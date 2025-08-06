import numpy as np
from light_gcn import EnhancedRecommendationSystem

def test_with_trained_model():
    print("=== Testing with Pre-Trained Model ===")
    try:
        loaded_model = EnhancedRecommendationSystem.load_model()
        print("✅ Pre-trained model loaded successfully!")
    except FileNotFoundError:
        print("❌ No trained model found! Please train the model first using CSV data.")
        return
    all_resources = [
        {'resource_id': 'R0001', 'resource_type': 'Motivational Message', 'content': 'You are not alone. Every step counts.'},
        {'resource_id': 'R0002', 'resource_type': 'Coping Strategy', 'content': 'Practice slow breathing when cravings hit.'},
        {'resource_id': 'R0003', 'resource_type': 'Article', 'content': 'Triggers are normal, but here\'s how to handle them.'},
        {'resource_id': 'R0004', 'resource_type': 'Motivational Message', 'content': 'Mistakes are part of growth. Keep pushing.'},
    ]
    all_users = [
        {'user_id': 'U0001', 'age_group': '18-24', 'gender': 'Male', 'recovery_stage': 'Early Stage', 'preferred_resource_type': 'Motivational Messages'},
        {'user_id': 'U0002', 'age_group': '25-34', 'gender': 'Female', 'recovery_stage': 'Mid Stage', 'preferred_resource_type': 'Coping Strategies'},
        {'user_id': 'U0003', 'age_group': '35-44', 'gender': 'Male', 'recovery_stage': 'Late Stage', 'preferred_resource_type': 'Articles'},
    ]
    user_interactions = [
        {'user_id': 'U0001', 'resource_id': 'R0003', 'interaction_type': 'dislike'},  # Dislike Article
        {'user_id': 'U0001', 'resource_id': 'R0002', 'interaction_type': 'like'},     # Like Coping Strategy
        {'user_id': 'U0002', 'resource_id': 'R0001', 'interaction_type': 'share'},
        {'user_id': 'U0003', 'resource_id': 'R0004', 'interaction_type': 'save'},
    ]
    print("\n=== SCENARIO 1: New User Recommendation ===")
    new_user_profile = {
        'user_id': 'U000x',
        'recovery_stage': 'Early Stage',
        'preferred_resource_type': 'Motivational Messages',
        'age_group': '18-24',
        'gender': 'Male'
    }
    user_specific_interactions = [
        interaction for interaction in user_interactions if interaction['user_id'] == new_user_profile['user_id']
    ]
    print(f"👤 User Profile: {new_user_profile['recovery_stage']}, prefers {new_user_profile['preferred_resource_type']}")
    print(f"📚 Available Resources: {len(all_resources)} resources")
    print(f"💭 User interaction history:")
    for interaction in user_specific_interactions:
        print(f"   - {interaction['resource_id']}: {interaction['interaction_type']}")
    recommendations = loaded_model.recommend_for_new_user(
        user_data=new_user_profile,
        available_resources=all_resources,
        user_interactions=user_specific_interactions,
        top_k=2
    )
    print(f"\n🎯 Top {len(recommendations)} Recommendations (should avoid Articles):")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. [{rec['resource_type']}] {rec['resource_id']}")
        print(f"   📝 {rec['content'][:70]}...")
        print(f"   📊 Score: {rec['recommendation_score']:.3f}")
    print("\n=== SCENARIO 2: Similar Users Recommendation ===")
    similar_users_result = loaded_model.find_similar_users_and_recommend(
        target_user_data=new_user_profile,
        all_users=all_users,
        available_resources=all_resources,
        user_interactions=user_interactions,
        top_k_users=2,
        top_k_resources=2
    )
    print(f"👤 Target User: {new_user_profile['user_id']}")
    print("Similar users:")
    for user in similar_users_result['similar_users']:
        print(f"   - User {user['user_id']} (Similarity: {user['similarity']:.4f})")
    print(f"\n🎯 Top {len(similar_users_result['recommendations'])} Recommendations from similar users:")
    for i, rec in enumerate(similar_users_result['recommendations'], 1):
        print(f"{i}. [{rec['resource_type']}] {rec['resource_id']}")
        print(f"   📝 {rec['content'][:70]}...")
        print(f"   📊 Score: {rec['recommendation_score']:.3f}")

def test_interaction_weight_impact():
    print("\n=== INTERACTION WEIGHT IMPACT TEST ===")
    try:
        loaded_model = EnhancedRecommendationSystem.load_model()
        base_user_profile = {
            'user_id': 'TEST_USER',
            'recovery_stage': 'Early Stage',
            'preferred_resource_type': 'Coping Strategies'
        }
        test_resources = [
            {'resource_id': 'TEST_RES_1', 'resource_type': 'Coping Strategy', 'content': 'Test resource 1'},
            {'resource_id': 'TEST_RES_2', 'resource_type': 'Article', 'content': 'Test resource 2'},
            {'resource_id': 'TEST_RES_3', 'resource_type': 'Article', 'content': 'Test resource 3'},
        ]
        interaction_types = ['like', 'dislike']
        print("🧪 Testing how different interaction types affect recommendations:")
        for interaction_type in interaction_types:
            print(f"\n--- User with '{interaction_type}' history on Article ---")
            interaction_history = [
                {'user_id': 'TEST_USER', 'resource_id': 'TEST_RES_2', 'interaction_type': interaction_type},
            ]
            recommendations = loaded_model.recommend_for_new_user(
                user_data=base_user_profile,
                available_resources=test_resources,
                user_interactions=interaction_history,
                top_k=3
            )
            avg_score = np.mean([rec['recommendation_score'] for rec in recommendations]) if recommendations else 0
            print(f"Average recommendation score: {avg_score:.3f}")
            for rec in recommendations:
                print(f"   - {rec['resource_id']} [{rec['resource_type']}]: {rec['recommendation_score']:.3f}")
    except Exception as e:
        print(f"❌ Error in interaction weight test: {e}")

def test_similar_users_recommendation():
    print("\n=== SIMILAR USERS RECOMMENDATION ===")
    try:
        loaded_model = EnhancedRecommendationSystem.load_model()
        target_user = {
            'user_id': 'U0004',
            'recovery_stage': 'Early Stage',
            'preferred_resource_type': 'Coping Strategies'
        }
        all_users = [
            {'user_id': 'U000x', 'recovery_stage': 'Early Stage', 'preferred_resource_type': 'Motivational Messages'},
            {'user_id': 'U0002', 'recovery_stage': 'Mid Stage', 'preferred_resource_type': 'Coping Strategies'},
            {'user_id': 'U0003', 'recovery_stage': 'Late Stage', 'preferred_resource_type': 'Articles'},
        ]
        resource_set = [
            {'resource_id': 'RES_A', 'resource_type': 'Coping Strategy', 'content': 'Strategy A'},
            {'resource_id': 'RES_B', 'resource_type': 'Coping Strategy', 'content': 'Strategy B'},
            {'resource_id': 'RES_C', 'resource_type': 'Motivational Message', 'content': 'Message C'},
            {'resource_id': 'RES_D', 'resource_type': 'Article', 'content': 'Article D'},
        ]
        user_interactions = [
            {'user_id': 'U0001', 'resource_id': 'RES_D', 'interaction_type': 'dislike'},  # Dislike Article
            {'user_id': 'U0002', 'resource_id': 'RES_A', 'interaction_type': 'share'},
            {'user_id': 'U0003', 'resource_id': 'RES_D', 'interaction_type': 'save'},
            {'user_id': 'U0004', 'resource_id': 'RES_D', 'interaction_type': 'dislike'},  # Target user dislikes Article
        ]
        result = loaded_model.find_similar_users_and_recommend(
            target_user_data=target_user,
            all_users=all_users,
            available_resources=resource_set,
            user_interactions=user_interactions,
            top_k_users=2,
            top_k_resources=2
        )
        print("👥 Similar users for target user:")
        for user in result['similar_users']:
            print(f"   - User {user['user_id']} (Similarity: {user['similarity']:.4f})")
        print("\n🎯 Recommendations based on similar users (should avoid Articles):")
        for rec in result['recommendations']:
            print(f"   - {rec['resource_id']} [{rec['resource_type']}]: {rec['recommendation_score']:.3f}")
    except Exception as e:
        print(f"❌ Error in similar users test: {e}")

def validate_recommendation_logic():
    print("\n=== RECOMMENDATION LOGIC VALIDATION ===")
    try:
        loaded_model = EnhancedRecommendationSystem.load_model()
        coping_user = {'user_id': 'TEST_USER', 'recovery_stage': 'Early Stage', 'preferred_resource_type': 'Coping Strategies'}
        test_resources = [
            {'resource_id': 'COPE_1', 'resource_type': 'Coping Strategy', 'content': 'Coping content'},
            {'resource_id': 'MOTIV_1', 'resource_type': 'Motivational Message', 'content': 'Motivational content'},
            {'resource_id': 'ART_1', 'resource_type': 'Article', 'content': 'Article content'}
        ]
        recommendations = loaded_model.recommend_for_new_user(
            user_data=coping_user,
            available_resources=test_resources,
            top_k=3
        )
        print("🧪 Logic Test - User prefers Coping Strategies:")
        print("Expected: Coping Strategy should rank highest")
        print("Actual results:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec['resource_type']} - Score: {rec['recommendation_score']:.3f}")
        if recommendations and recommendations[0]['resource_type'] == 'Coping Strategy':
            print("✅ PASS: Coping Strategy ranked highest")
        else:
            print("⚠️  REVIEW: Expected Coping Strategy to rank highest")
    except Exception as e:
        print(f"❌ Validation error: {e}")

def validate_dislike_impact():
    print("\n=== DISLIKE IMPACT VALIDATION ===")
    try:
        loaded_model = EnhancedRecommendationSystem.load_model()
        test_user = {'user_id': 'TEST_USER', 'recovery_stage': 'Early Stage', 'preferred_resource_type': 'Motivational Messages'}
        test_resources = [
            {'resource_id': 'RES_1', 'resource_type': 'Article', 'content': 'Article content 1'},
            {'resource_id': 'RES_2', 'resource_type': 'Article', 'content': 'Article content 2'},
            {'resource_id': 'RES_3', 'resource_type': 'Motivational Message', 'content': 'Motivational content'},
        ]
        interactions = [
            {'user_id': 'TEST_USER', 'resource_id': 'RES_1', 'interaction_type': 'dislike'},
        ]
        recommendations = loaded_model.recommend_for_new_user(
            user_data=test_user,
            available_resources=test_resources,
            user_interactions=interactions,
            top_k=2
        )
        print("🧪 Dislike Test - User dislikes an Article:")
        print("Expected: Motivational Message should rank higher than Articles")
        print("Actual results:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec['resource_type']} - Score: {rec['recommendation_score']:.3f}")
        if recommendations and recommendations[0]['resource_type'] != 'Article':
            print("✅ PASS: Non-Article ranked highest")
        else:
            print("⚠️  REVIEW: Expected non-Article to rank highest")
    except Exception as e:
        print(f"❌ Dislike impact validation error: {e}")

if __name__ == "__main__":
    print("🚀 Testing Recommendation System with Pre-Trained Model")
    print("=" * 60)
    test_with_trained_model()
    test_interaction_weight_impact()
    test_similar_users_recommendation()
    validate_recommendation_logic()
    validate_dislike_impact()
    print("\n" + "=" * 60)
    print("✅ Testing completed!")
