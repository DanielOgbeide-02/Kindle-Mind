from flask import Flask, request, jsonify
import logging
from light_gcn import EnhancedRecommendationSystem


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variable to store the recommendation system
rec_system = None

def load_recommendation_system(model_path="lightgcn_model.pt", mappings_path="mappings.pkl"):
    """Load the pre-trained recommendation system."""
    try:
        logger.info("Loading pre-trained model...")
        rec_system = EnhancedRecommendationSystem.load_model(model_path, mappings_path)
        logger.info("Model loaded successfully!")
        return rec_system
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

# Initialize the recommendation system at startup
try:
    rec_system = load_recommendation_system()
except Exception as e:
    logger.error(f"Failed to initialize application: {e}")
    exit(1)

@app.route("/recommend/new_user", methods=["POST"])
def recommend_new_user():
    """API endpoint to recommend resources for a new user."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Validate required fields
        required_fields = ["recovery_stage", "preferred_resource_type", "available_resources"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        user_data = {
            "user_id": data.get("user_id", "new_user"),
            "recovery_stage": data["recovery_stage"],
            "preferred_resource_type": data["preferred_resource_type"]
        }
        available_resources = data["available_resources"]
        interactions = data.get("interactions", [])
        top_k = data.get("top_k", 5)

        # Validate available_resources format
        for resource in available_resources:
            if not isinstance(resource, dict) or "resource_id" not in resource or "resource_type" not in resource:
                return jsonify({"error": "Invalid resource format. Each resource must have 'resource_id' and 'resource_type'"}), 400

        # Validate interactions format
        for interaction in interactions:
            if not isinstance(interaction, dict) or "resource_id" not in interaction or "interaction_type" not in interaction:
                return jsonify({"error": "Invalid interaction format. Each interaction must have 'resource_id' and 'interaction_type'"}), 400

        recommendations = rec_system.recommend_for_new_user(
            user_data=user_data,
            available_resources=available_resources,
            user_interactions=interactions,
            top_k=top_k
        )

        return jsonify({
            "status": "success",
            "recommended_resources": recommendations
        }), 200

    except Exception as e:
        logger.error(f"Error in recommend_new_user: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/recommend/similar_users", methods=["POST"])
def recommend_similar_users():
    """API endpoint to find similar users only (no resource recommendations)."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Validate required fields
        required_fields = ["user_data", "all_users"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        user_data = data["user_data"]
        all_users = data["all_users"]
        interactions = data.get("interactions", [])
        top_k_users = data.get("top_k_users", 3)

        # Validate user_data format
        if not isinstance(user_data, dict) or "recovery_stage" not in user_data or "preferred_resource_type" not in user_data:
            return jsonify({"error": "Invalid user_data format. Must include 'recovery_stage' and 'preferred_resource_type'"}), 400

        # Validate all_users format
        for user in all_users:
            if not isinstance(user, dict) or "user_id" not in user or "recovery_stage" not in user or "preferred_resource_type" not in user:
                return jsonify({"error": "Invalid user format. Each user must have 'user_id', 'recovery_stage', and 'preferred_resource_type'"}), 400

        # Validate interactions format
        for interaction in interactions:
            if not isinstance(interaction, dict) or "resource_id" not in interaction or "interaction_type" not in interaction:
                return jsonify({"error": "Invalid interaction format. Each interaction must have 'resource_id' and 'interaction_type'"}), 400

        # ✅ Use the new similar users-only method
        result = rec_system.find_similar_users(
            target_user_data=user_data,
            all_users=all_users,
            user_interactions=interactions,
            top_k_users=top_k_users
        )

        return jsonify({
            "status": "success",
            "similar_users": result["similar_users"]
        }), 200

    except Exception as e:
        logger.error(f"Error in recommend_similar_users: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify the API is running."""
    return jsonify({"status": "API is running", "model_loaded": rec_system is not None}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
