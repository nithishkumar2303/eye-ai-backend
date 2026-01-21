from flask import Blueprint, request, jsonify
from .models import db, Prediction
from .auth import token_required
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import logging

prediction_bp = Blueprint('predictions', __name__, url_prefix='/predictions')
logger = logging.getLogger(__name__)

# Import your existing model loading code
from main import model, model_loaded, device, transform, CLASS_LABELS, GRADE_DESCRIPTIONS


@prediction_bp.route('/predict', methods=['POST'])
@token_required
def predict(current_user):
    """Make a prediction and save to history"""
    try:
        if not model_loaded:
            return jsonify({'message': 'Model not loaded'}), 503
        
        if 'file' not in request.files:
            return jsonify({'message': 'No file provided'}), 400
        
        file = request. files['file']
        
        if file.filename == '':
            return jsonify({'message': 'No file selected'}), 400
        
        if not file.content_type or not file.content_type.startswith('image/'):
            return jsonify({'message': 'File must be an image'}), 400
        
        # Read and process image
        image_bytes = file.read()
        
        if len(image_bytes) > 10 * 1024 * 1024:  # 10MB limit
            return jsonify({'message': 'File too large (max 10MB)'}), 413
        
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception: 
            return jsonify({'message':  'Invalid image file'}), 400
        
        # Make prediction
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch. no_grad():
            outputs = model(input_tensor)
            logits = outputs[0] if outputs. dim() == 2 else outputs. squeeze()
            probabilities = F.softmax(logits, dim=0)
            predicted_class = int(torch.argmax(logits).item())
            confidence = float(probabilities[predicted_class]. item())
        
        # Prepare response
        response_data = {
            'success': True,
            'predicted_grade': predicted_class,
            'predicted_class':  CLASS_LABELS[predicted_class],
            'description': GRADE_DESCRIPTIONS[predicted_class],
            'confidence': round(confidence * 100, 2),
            'probabilities': {
                f'grade_{i}': round(prob. item() * 100, 2)
                for i, prob in enumerate(probabilities)
            },
            'all_grades': [
                {
                    'grade': i,
                    'label': CLASS_LABELS[i],
                    'description': GRADE_DESCRIPTIONS[i],
                    'probability': round(probabilities[i]. item() * 100, 2)
                }
                for i in range(4)
            ]
        }
        
        # Save prediction to database
        try:
            prediction = Prediction(
                user_id=current_user.id,
                predicted_grade=predicted_class,
                predicted_class=CLASS_LABELS[predicted_class],
                confidence=confidence,
                description=GRADE_DESCRIPTIONS[predicted_class],
                probabilities=response_data['probabilities'],
                image_filename=file. filename
            )
            
            db.session.add(prediction)
            db.session.commit()
            
            response_data['prediction_id'] = prediction.id
        except Exception as e:
            logger.error(f"Failed to save prediction: {e}")
            db.session.rollback()
            # Still return the prediction, but note the database error
            response_data['warning'] = 'Prediction completed but not saved to history'
        
        return jsonify(response_data), 200
    
    except Exception as e: 
        logger.error(f"Prediction error: {e}")
        return jsonify({'message': f'Prediction error: {str(e)}'}), 500


@prediction_bp.route('/history', methods=['GET'])
@token_required
def get_history(current_user):
    """Get user's prediction history"""
    try: 
        page = request.args.get('page', 1, type=int)
        per_page = request. args.get('per_page', 10, type=int)
        
        predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(
            Prediction.created_at.desc()
        ).paginate(page=page, per_page=per_page)
        
        return jsonify({
            'predictions': [p.to_dict() for p in predictions.items],
            'total': predictions.total,
            'pages': predictions.pages,
            'current_page': page
        }), 200
    
    except Exception as e: 
        logger.error(f"Error fetching history: {e}")
        return jsonify({'message': f'Error:  {str(e)}'}), 500


@prediction_bp.route('/<prediction_id>', methods=['GET'])
@token_required
def get_prediction(prediction_id, current_user):
    """Get a specific prediction"""
    try: 
        prediction = Prediction.query. filter_by(
            id=prediction_id,
            user_id=current_user.id
        ).first()
        
        if not prediction: 
            return jsonify({'message':  'Prediction not found'}), 404
        
        return jsonify({'prediction': prediction.to_dict()}), 200
    
    except Exception as e:
        logger. error(f"Error fetching prediction: {e}")
        return jsonify({'message': f'Error:  {str(e)}'}), 500


@prediction_bp.route('/<prediction_id>', methods=['DELETE'])
@token_required
def delete_prediction(prediction_id, current_user):
    """Delete a prediction"""
    try: 
        prediction = Prediction.query. filter_by(
            id=prediction_id,
            user_id=current_user.id
        ).first()
        
        if not prediction:
            return jsonify({'message': 'Prediction not found'}), 404
        
        db.session.delete(prediction)
        db.session. commit()
        
        return jsonify({'message': 'Prediction deleted successfully'}), 200
    
    except Exception as e: 
        logger.error(f"Error deleting prediction: {e}")
        db.session.rollback()
        return jsonify({'message': f'Error: {str(e)}'}), 500