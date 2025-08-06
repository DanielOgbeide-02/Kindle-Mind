
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.modules.sparse
import torch.nn.modules.linear
import torch.serialization
from typing import Dict, List, Tuple, Optional, Union
import logging
import pickle
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LightGCNWithUserAndItemInfo(nn.Module):
    def __init__(self, num_users, num_items, recovery_vocab, type_vocab, resource_type_vocab,
                 embedding_dim=64, feature_dim=8, num_layers=3):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.recovery_embedding = nn.Embedding(recovery_vocab, feature_dim)
        self.type_embedding = nn.Embedding(type_vocab, feature_dim)
        self.resource_type_embedding = nn.Embedding(resource_type_vocab, feature_dim)
        self.user_project = nn.Linear(embedding_dim + 2 * feature_dim, embedding_dim)
        self.item_project = nn.Linear(embedding_dim + feature_dim, embedding_dim)
        self.num_layers = num_layers
        self.num_users = num_users
        self.num_items = num_items
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.recovery_embedding.weight)
        nn.init.xavier_uniform_(self.type_embedding.weight)
        nn.init.xavier_uniform_(self.resource_type_embedding.weight)
        nn.init.xavier_uniform_(self.user_project.weight)
        nn.init.xavier_uniform_(self.item_project.weight)

    def forward(self, adj, recovery_stage_idx, preferred_type_idx, resource_type_idx):
        device = self.user_embedding.weight.device
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        recovery_emb = self.recovery_embedding(torch.tensor(recovery_stage_idx, device=device))
        type_emb = self.type_embedding(torch.tensor(preferred_type_idx, device=device))
        resource_type_emb = self.resource_type_embedding(torch.tensor(resource_type_idx, device=device))
        enriched_user_emb = self.user_project(torch.cat([user_emb, recovery_emb, type_emb], dim=1))
        enriched_item_emb = self.item_project(torch.cat([item_emb, resource_type_emb], dim=1))
        user_embs = [enriched_user_emb]
        item_embs = [enriched_item_emb]
        for _ in range(self.num_layers):
            user_emb = torch.matmul(adj, item_embs[-1])
            item_emb = torch.matmul(adj.T, user_embs[-1])
            user_embs.append(user_emb)
            item_embs.append(item_emb)
        return torch.stack(user_embs, dim=1).mean(dim=1), torch.stack(item_embs, dim=1).mean(dim=1)

    def get_embedding_for_new_user(self, recovery_stage: str, preferred_type: str,
                                   recovery_stage_map: Dict, preferred_type_map: Dict):
        device = self.user_embedding.weight.device
        if recovery_stage not in recovery_stage_map:
            recovery_idx = 0
        else:
            recovery_idx = recovery_stage_map[recovery_stage]
        if preferred_type not in preferred_type_map:
            preferred_idx = 0
        else:
            preferred_idx = preferred_type_map[preferred_type]
        recovery_emb = self.recovery_embedding(torch.tensor([recovery_idx], device=device))
        type_emb = self.type_embedding(torch.tensor([preferred_idx], device=device))
        avg_user_emb = self.user_embedding.weight.mean(dim=0, keepdim=True)
        combined_emb = torch.cat([avg_user_emb, recovery_emb, type_emb], dim=1)
        user_emb = self.user_project(combined_emb)
        return user_emb.squeeze(0)

class EnhancedRecommendationSystem:
    def __init__(self, embedding_dim=64, feature_dim=8, num_layers=3, epochs=50, lr=0.01):
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.model = None
        self.is_trained = False

    def load_data_from_csv(self, users_csv: str, resources_csv: str, interactions_csv: str) -> Dict:
        logger.info("Loading data from CSV files...")
        users_df = pd.read_csv(users_csv)
        resources_df = pd.read_csv(resources_csv)
        interactions_df = pd.read_csv(interactions_csv)
        return self.prepare_data_from_dataframes(users_df, resources_df, interactions_df)

    def prepare_data_from_dataframes(self, users_df: pd.DataFrame, resources_df: pd.DataFrame,
                                     interactions_df: pd.DataFrame) -> Dict:
        logger.info(f"Preparing data: {len(users_df)} users, {len(resources_df)} resources, {len(interactions_df)} interactions")
        positive_interactions = ['like', 'share', 'save', 'view']
        negative_interactions = ['dislike']
        positive_df = interactions_df[interactions_df['interaction_type'].isin(positive_interactions)].copy()
        negative_df = interactions_df[interactions_df['interaction_type'].isin(negative_interactions)].copy()
        if len(positive_df) == 0:
            logger.warning("No positive interactions found!")
            return None
        user_id_map = {id_: idx for idx, id_ in enumerate(users_df['user_id'].unique())}
        resource_id_map = {id_: idx for idx, id_ in enumerate(resources_df['resource_id'].unique())}
        positive_df['user_index'] = positive_df['user_id'].map(user_id_map)
        positive_df['resource_index'] = positive_df['resource_id'].map(resource_id_map)
        negative_df['user_index'] = negative_df['user_id'].map(user_id_map)
        negative_df['resource_index'] = negative_df['resource_id'].map(resource_id_map)
        positive_df = positive_df.dropna(subset=['user_index', 'resource_index'])
        negative_df = negative_df.dropna(subset=['user_index', 'resource_index'])
        positive_df['user_index'] = positive_df['user_index'].astype(int)
        positive_df['resource_index'] = positive_df['resource_index'].astype(int)
        negative_df['user_index'] = negative_df['user_index'].astype(int)
        negative_df['resource_index'] = negative_df['resource_index'].astype(int)
        num_users = len(user_id_map)
        num_items = len(resource_id_map)
        interaction_matrix = np.zeros((num_users, num_items), dtype=np.float32)
        for _, row in positive_df.iterrows():
            interaction_matrix[row['user_index'], row['resource_index']] = 1.0
        for _, row in negative_df.iterrows():
            interaction_matrix[row['user_index'], row['resource_index']] = -1.0
        recovery_stages = users_df['recovery_stage'].unique()
        preferred_types = users_df['preferred_resource_type'].unique()
        recovery_stage_map = {v: i for i, v in enumerate(recovery_stages)}
        preferred_type_map = {v: i for i, v in enumerate(preferred_types)}
        users_df['recovery_stage_idx'] = users_df['recovery_stage'].map(recovery_stage_map)
        users_df['preferred_type_idx'] = users_df['preferred_resource_type'].map(preferred_type_map)
        user_map_df = pd.DataFrame(user_id_map.items(), columns=["user_id", "user_index"])
        user_map_df = user_map_df.sort_values('user_index')
        recovery_stage_indices = users_df.set_index('user_id').loc[user_map_df['user_id']]['recovery_stage_idx'].values
        preferred_type_indices = users_df.set_index('user_id').loc[user_map_df['user_id']]['preferred_type_idx'].values
        resource_types = resources_df['resource_type'].unique()
        resource_type_map = {v: i for i, v in enumerate(resource_types)}
        resources_df['resource_type_idx'] = resources_df['resource_type'].map(resource_type_map)
        resource_map_df = pd.DataFrame(resource_id_map.items(), columns=["resource_id", "resource_index"])
        resource_map_df = resource_map_df.sort_values('resource_index')
        resource_type_indices = resources_df.set_index('resource_id').loc[resource_map_df['resource_id']]['resource_type_idx'].values
        return {
            'interaction_matrix': interaction_matrix,
            'positive_df': positive_df,
            'negative_df': negative_df,
            'user_id_map': user_id_map,
            'resource_id_map': resource_id_map,
            'recovery_stage_indices': recovery_stage_indices,
            'preferred_type_indices': preferred_type_indices,
            'resource_type_indices': resource_type_indices,
            'recovery_stage_map': recovery_stage_map,
            'preferred_type_map': preferred_type_map,
            'resource_type_map': resource_type_map,
            'users_df': users_df,
            'resources_df': resources_df,
            'num_users': num_users,
            'num_items': num_items
        }

    def train_model(self, data_dict: Dict):
        logger.info("Starting model training...")
        interaction_matrix = data_dict['interaction_matrix']
        positive_df = data_dict['positive_df']
        negative_df = data_dict['negative_df']
        recovery_stage_indices = data_dict['recovery_stage_indices']
        preferred_type_indices = data_dict['preferred_type_indices']
        resource_type_indices = data_dict['resource_type_indices']
        adj = torch.tensor(interaction_matrix)
        adj_norm = torch.where(adj != 0, 1 / (torch.abs(adj).sum(dim=1, keepdim=True) + 1e-10), torch.tensor(0.0))
        adj_norm = adj_norm.float()
        self.model = LightGCNWithUserAndItemInfo(
            num_users=data_dict['num_users'],
            num_items=data_dict['num_items'],
            recovery_vocab=len(data_dict['recovery_stage_map']),
            type_vocab=len(data_dict['preferred_type_map']),
            resource_type_vocab=len(data_dict['resource_type_map']),
            embedding_dim=self.embedding_dim,
            feature_dim=self.feature_dim,
            num_layers=self.num_layers
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        for epoch in range(self.epochs):
            self.model.train()
            user_emb, item_emb = self.model(
                adj_norm,
                recovery_stage_indices,
                preferred_type_indices,
                resource_type_indices
            )
            pos_u = torch.tensor(positive_df['user_index'].values)
            pos_i = torch.tensor(positive_df['resource_index'].values)
            neg_i = torch.randint(0, data_dict['num_items'], pos_i.shape)
            if len(negative_df) > 0:
                neg_u = torch.tensor(negative_df['user_index'].values)
                neg_i_dislike = torch.tensor(negative_df['resource_index'].values)
                neg_i = torch.cat([neg_i, neg_i_dislike])
                pos_u = torch.cat([pos_u, neg_u])
                pos_i = torch.cat([pos_i, neg_i_dislike])
                labels = torch.cat([torch.ones(len(positive_df)), -torch.ones(len(negative_df))])
                scores = (user_emb[pos_u] * item_emb[pos_i]).sum(dim=1)
                loss = -torch.log(torch.sigmoid(scores * labels)).mean()
            else:
                pos_scores = (user_emb[pos_u] * item_emb[pos_i]).sum(dim=1)
                neg_scores = (user_emb[pos_u] * item_emb[neg_i]).sum(dim=1)
                loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
        self.model.eval()
        with torch.no_grad():
            final_user_emb, final_item_emb = self.model(
                adj_norm,
                recovery_stage_indices,
                preferred_type_indices,
                resource_type_indices
            )
            self.model.final_item_emb = final_item_emb
            self.model.final_user_emb = final_user_emb
        self.data_dict = data_dict
        self.adj_norm = adj_norm
        self.is_trained = True
        logger.info("Model training completed!")

    def recommend_for_new_user(self, user_data: Dict, available_resources: List[Dict],
                              user_interactions: Union[List[str], List[Dict]] = None, top_k: int = 5) -> List[Dict]:
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        self.model.eval()
        with torch.no_grad():
            user_emb = self.model.get_embedding_for_new_user(
                user_data['recovery_stage'],
                user_data['preferred_resource_type'],
                self.data_dict['recovery_stage_map'],
                self.data_dict['preferred_type_map']
            )
            resource_embeddings = []
            resource_info = []
            resource_types = []
            for resource in available_resources:
                resource_type = resource['resource_type']
                resource_types.append(resource_type)
                if resource_type in self.data_dict['resource_type_map']:
                    resource_type_idx = self.data_dict['resource_type_map'][resource_type]
                else:
                    resource_type_idx = 0
                resource_type_emb = self.model.resource_type_embedding(
                    torch.tensor([resource_type_idx])
                )
                avg_item_emb = self.model.item_embedding.weight.mean(dim=0, keepdim=True)
                combined_emb = torch.cat([avg_item_emb, resource_type_emb], dim=1)
                resource_emb = self.model.item_project(combined_emb)
                resource_embeddings.append(resource_emb.squeeze(0))
                resource_info.append(resource)
            if not resource_embeddings:
                return []
            resource_emb_tensor = torch.stack(resource_embeddings)
            scores = torch.matmul(resource_emb_tensor, user_emb)
            disliked_types = set()
            interacted_resource_ids = set()
            if user_interactions:
                for interaction in user_interactions:
                    if isinstance(interaction, dict):
                        resource_id = interaction['resource_id']
                        interaction_type = interaction['interaction_type']
                        interacted_resource_ids.add(resource_id)
                        if interaction_type == 'dislike':
                            for res in available_resources:
                                if res['resource_id'] == resource_id:
                                    disliked_types.add(res['resource_type'])
                                    break
                    else:
                        interacted_resource_ids.add(interaction)
            for i, (resource, res_type) in enumerate(zip(resource_info, resource_types)):
                if resource['resource_id'] in interacted_resource_ids:
                    scores[i] = -float('inf')
                elif res_type in disliked_types:
                    scores[i] *= 0.5
            top_scores, top_indices = torch.topk(scores, min(top_k, len(scores)))
            recommendations = []
            for idx, score in zip(top_indices, top_scores):
                if score.item() != -float('inf'):
                    resource = resource_info[idx.item()].copy()
                    resource['recommendation_score'] = score.item()
                    recommendations.append(resource)
            return recommendations

    def find_similar_users_and_recommend(self, target_user_data: Dict, all_users: List[Dict],
                                        available_resources: List[Dict], user_interactions: Union[List[str], List[Dict]] = None,
                                        top_k_users: int = 3, top_k_resources: int = 5) -> Dict:
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        self.model.eval()
        with torch.no_grad():
            target_emb = self.model.get_embedding_for_new_user(
                target_user_data['recovery_stage'],
                target_user_data['preferred_resource_type'],
                self.data_dict['recovery_stage_map'],
                self.data_dict['preferred_type_map']
            ).cpu().numpy().reshape(1, -1)
            user_embeddings = []
            user_ids = []
            for user in all_users:
                if user['user_id'] != target_user_data.get('user_id'):
                    emb = self.model.get_embedding_for_new_user(
                        user['recovery_stage'],
                        user['preferred_resource_type'],
                        self.data_dict['recovery_stage_map'],
                        self.data_dict['preferred_type_map']
                    ).cpu().numpy()
                    user_embeddings.append(emb)
                    user_ids.append(user['user_id'])
            if not user_embeddings:
                return {'similar_users': [], 'recommendations': []}
            user_emb_tensor = np.stack(user_embeddings)
            similarities = cosine_similarity(target_emb, user_emb_tensor)[0]
            top_indices = np.argsort(similarities)[-top_k_users:][::-1]
            similar_users = [
                {'user_id': user_ids[i], 'similarity': float(similarities[i])}
                for i in top_indices
            ]
            similar_user_ids = [user['user_id'] for user in similar_users]
            interaction_scores = {}
            disliked_types = set()
            interacted_resource_ids = set()
            for resource in available_resources:
                interaction_scores[resource['resource_id']] = 0.0
            if user_interactions:
                for interaction in user_interactions:
                    if isinstance(interaction, dict):
                        resource_id = interaction['resource_id']
                        user_id = interaction.get('user_id')
                        interaction_type = interaction['interaction_type']
                        if user_id == target_user_data.get('user_id'):
                            interacted_resource_ids.add(resource_id)
                            if interaction_type == 'dislike':
                                for res in available_resources:
                                    if res['resource_id'] == resource_id:
                                        disliked_types.add(res['resource_type'])
                                        break
                        if user_id in similar_user_ids:
                            if interaction_type in ['like', 'share', 'save']:
                                interaction_scores[resource_id] += 1.0
                            elif interaction_type == 'dislike':
                                interaction_scores[resource_id] -= 0.5
                    else:
                        if target_user_data.get('user_id'):
                            interacted_resource_ids.add(interaction)
            resource_scores = [
                {'resource_id': rid, 'score': score}
                for rid, score in interaction_scores.items()
                if rid not in interacted_resource_ids
            ]
            for res_score in resource_scores:
                for res in available_resources:
                    if res['resource_id'] == res_score['resource_id'] and res['resource_type'] in disliked_types:
                        res_score['score'] *= 0.5
                        break
            resource_scores.sort(key=lambda x: x['score'], reverse=True)
            top_resource_ids = [rs['resource_id'] for rs in resource_scores[:top_k_resources] if rs['score'] > 0]
            recommendations = [
                resource.copy() for resource in available_resources
                if resource['resource_id'] in top_resource_ids
            ]
            for rec in recommendations:
                rec['recommendation_score'] = interaction_scores[rec['resource_id']]
            return {
                'similar_users': similar_users,
                'recommendations': recommendations
            }

    def recommend_for_user_by_id(self, user_id: str, top_k: int = 5) -> Tuple[List[str], List[float]]:
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        if user_id not in self.data_dict['user_id_map']:
            logger.warning(f"User {user_id} not found in training data. Use recommend_for_new_user() instead.")
            return [], []
        user_index = self.data_dict['user_id_map'][user_id]
        interaction_matrix = csr_matrix(self.data_dict['interaction_matrix'])
        user_interaction_indices = interaction_matrix[user_index].nonzero()[1]
        user_emb = self.model.final_user_emb[user_index] if hasattr(self.model, 'final_user_emb') else None
        if user_emb is None:
            logger.error("Model not properly trained - missing final embeddings")
            return [], []
        item_emb = self.model.final_item_emb
        scores = torch.matmul(item_emb, user_emb)
        if len(user_interaction_indices) > 0:
            scores[user_interaction_indices] = -float('inf')
        top_scores, top_indices = torch.topk(scores, min(top_k, len(scores)))
        resource_id_reverse_map = {v: k for k, v in self.data_dict['resource_id_map'].items()}
        recommended_resource_ids = [resource_id_reverse_map[idx.item()] for idx in top_indices]
        return recommended_resource_ids, top_scores.tolist()

    def save_model(self, model_path: str = "lightgcn_model.pt", mappings_path: str = "mappings.pkl"):
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        torch.save(self.model, model_path)
        mappings_data = {
            'recovery_stage_map': self.data_dict['recovery_stage_map'],
            'preferred_type_map': self.data_dict['preferred_type_map'],
            'resource_type_map': self.data_dict['resource_type_map'],
            'num_users': self.data_dict['num_users'],
            'num_items': self.data_dict['num_items'],
            'user_id_map': self.data_dict['user_id_map'],
            'resource_id_map': self.data_dict['resource_id_map'],
        }
        with open(mappings_path, 'wb') as f:
            pickle.dump(mappings_data, f)
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Mappings saved to {mappings_path}")

    @classmethod
    def load_model(cls, model_path: str = "lightgcn_model.pt", mappings_path: str = "mappings.pkl"):
        # Allowlist the custom class and PyTorch modules for safe loading
        torch.serialization.add_safe_globals([
            LightGCNWithUserAndItemInfo,
            torch.nn.modules.sparse.Embedding,
            torch.nn.modules.linear.Linear
        ])
        rec_system = cls()
        try:
            rec_system.model = torch.load(model_path, map_location=torch.device("cpu"))
            rec_system.model.eval()
            with open(mappings_path, 'rb') as f:
                mappings_data = pickle.load(f)
            rec_system.data_dict = mappings_data
            rec_system.is_trained = True
            logger.info("Model and mappings loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        return rec_system

def train_and_save_model(users_csv: str, resources_csv: str, interactions_csv: str):
    print("Loading data from CSV files...")
    rec_system = EnhancedRecommendationSystem(epochs=30)
    print("Preparing data...")
    data_dict = rec_system.load_data_from_csv(users_csv, resources_csv, interactions_csv)
    if data_dict is None:
        print("Error: No valid data loaded from CSV files. Training aborted.")
        return None
    print("Training model...")
    rec_system.train_model(data_dict)
    print("Saving model...")
    rec_system.save_model("lightgcn_model.pt", "mappings.pkl")
    print("Training complete! Model saved.")
    return rec_system

def test_new_user_recommendations():
    print("=" * 50)
    print("TESTING NEW USER RECOMMENDATIONS")
    print("=" * 50)
    try:
        rec_system = EnhancedRecommendationSystem.load_model("lightgcn_model.pt", "mappings.pkl")
    except FileNotFoundError:
        print("Trained model not found. Please train the model first.")
        return
    resources = [
        {"resource_id": "R0001", "resource_type": "Motivational Message", "content": "You are not alone. Every step counts."},
        {"resource_id": "R0002", "resource_type": "Coping Strategy", "content": "Practice slow breathing when cravings hit."},
        {"resource_id": "R0003", "resource_type": "Article", "content": "Triggers are normal, but here's how to handle them."},
        {"resource_id": "R0004", "resource_type": "Motivational Message", "content": "Mistakes are part of growth. Keep pushing."},
    ]
    all_users = [
        {"user_id": "U0001", "age_group": "18-24", "gender": "Male", "recovery_stage": "Early Stage", "preferred_resource_type": "Motivational Messages"},
        {"user_id": "U0002", "age_group": "25-34", "gender": "Female", "recovery_stage": "Mid Stage", "preferred_resource_type": "Coping Strategies"},
        {"user_id": "U0003", "age_group": "35-44", "gender": "Male", "recovery_stage": "Late Stage", "preferred_resource_type": "Articles"},
    ]
    print("\nTest Case 1: New user with interactions including dislike")
    new_user = {
        'user_id': 'U0001',
        'recovery_stage': 'Early Stage',
        'preferred_resource_type': 'Motivational Messages'
    }
    interactions = [
        {'user_id': 'U0001', 'resource_id': 'R0003', 'interaction_type': 'dislike'},  # Dislike an Article
        {'user_id': 'U0001', 'resource_id': 'R0002', 'interaction_type': 'like'},     # Like a Coping Strategy
    ]
    recommendations = rec_system.recommend_for_new_user(
        user_data=new_user,
        available_resources=resources,
        user_interactions=interactions,
        top_k=2
    )
    print("Recommendations for new user (should avoid Articles):")
    for rec in recommendations:
        print(f"- {rec['resource_id']}: {rec['content']} (Type: {rec['resource_type']}, Score: {rec['recommendation_score']:.4f})")
    print("\nTest Case 2: Find similar users and recommend")
    interactions_all = interactions + [
        {'user_id': 'U0002', 'resource_id': 'R0001', 'interaction_type': 'share'},
        {'user_id': 'U0003', 'resource_id': 'R0004', 'interaction_type': 'save'},
    ]
    similar_users_result = rec_system.find_similar_users_and_recommend(
        target_user_data=new_user,
        all_users=all_users,
        available_resources=resources,
        user_interactions=interactions_all,
        top_k_users=2,
        top_k_resources=2
    )
    print("Similar users:")
    for user in similar_users_result['similar_users']:
        print(f"- User {user['user_id']} (Similarity: {user['similarity']:.4f})")
    print("Recommendations based on similar users (should avoid target user's disliked types):")
    for rec in similar_users_result['recommendations']:
        print(f"- {rec['resource_id']}: {rec['content']} (Type: {rec['resource_type']}, Score: {rec['recommendation_score']:.4f})")
    print("\nTesting complete!")

if __name__ == "__main__":
    rec_system = train_and_save_model("expanded_users.csv", "expanded_resources.csv", "expanded_interactions.csv")
    if rec_system:
        test_new_user_recommendations()
