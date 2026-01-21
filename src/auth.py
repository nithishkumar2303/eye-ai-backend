import jwt
import os
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify, current_app
from . models import User

class AuthUtils:
    """Utilities for JWT token generation and verification"""
    
    @staticmethod
    def generate_token(user_id:  str, expires_in_hours: int = 24) -> str:
        """Generate JWT token for user"""
        payload = {
            'user_id':  user_id,
            'exp': datetime.utcnow() + timedelta(hours=expires_in_hours),
            'iat': datetime. utcnow()
        }
        token = jwt.encode(
            payload,
            current_app.config['SECRET_KEY'],
            algorithm='HS256'
        )
        return token
    
    @staticmethod
    def verify_token(token: str) -> dict | None:
        """Verify JWT token and return payload"""
        try:
            payload = jwt.decode(
                token,
                current_app.config['SECRET_KEY'],
                algorithms=['HS256']
            )
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt. InvalidTokenError:
            return None
    
    @staticmethod
    def get_current_user(token: str) -> User | None:
        """Get user from token"""
        payload = AuthUtils.verify_token(token)
        if not payload:
            return None
        
        user = User.query.get(payload['user_id'])
        return user


def token_required(f):
    """Decorator to require valid JWT token"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Check if token is in headers
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(" ")[1]
            except IndexError:
                return jsonify({'message': 'Invalid token format'}), 401
        
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        
        user = AuthUtils.get_current_user(token)
        if not user:
            return jsonify({'message': 'Invalid or expired token'}), 401
        
        # Pass user to the route function
        kwargs['current_user'] = user
        return f(*args, **kwargs)
    
    return decorated