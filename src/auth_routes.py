from flask import Blueprint, request, jsonify
from .models import db, User
from .auth import AuthUtils
import re

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

# Email validation regex
EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')


def validate_email(email: str) -> bool:
    """Validate email format"""
    return EMAIL_REGEX.match(email) is not None


def validate_password(password: str) -> tuple[bool, str]:
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one digit"
    return True, "Password is valid"


def validate_username(username: str) -> tuple[bool, str]:
    """Validate username"""
    if len(username) < 3:
        return False, "Username must be at least 3 characters"
    if len(username) > 20:
        return False, "Username must be less than 20 characters"
    if not username.replace('_', '').replace('-', '').isalnum():
        return False, "Username can only contain letters, numbers, hyphens, and underscores"
    return True, "Username is valid"


@auth_bp. route('/register', methods=['POST'])
def register():
    """Register a new user"""
    try:
        data = request.get_json()
        
        # Validate input
        if not data: 
            return jsonify({'message':  'No data provided'}), 400
        
        email = data.get('email', '').strip().lower()
        username = data. get('username', '').strip()
        password = data.get('password', '')
        
        # Email validation
        if not email or not validate_email(email):
            return jsonify({'message':  'Invalid email format'}), 400
        
        # Username validation
        is_valid, msg = validate_username(username)
        if not is_valid: 
            return jsonify({'message': msg}), 400
        
        # Password validation
        is_valid, msg = validate_password(password)
        if not is_valid: 
            return jsonify({'message': msg}), 400
        
        # Check if user already exists
        if User.query.filter_by(email=email).first():
            return jsonify({'message': 'Email already registered'}), 409
        
        if User.query.filter_by(username=username).first():
            return jsonify({'message': 'Username already taken'}), 409
        
        # Create new user
        user = User(email=email, username=username)
        user.set_password(password)
        
        db. session.add(user)
        db.session.commit()
        
        # Generate token
        token = AuthUtils.generate_token(user.id)
        
        return jsonify({
            'message': 'User registered successfully',
            'user': user.to_dict(),
            'token': token
        }), 201
    
    except Exception as e: 
        db.session.rollback()
        return jsonify({'message':  f'Registration error: {str(e)}'}), 500


@auth_bp. route('/login', methods=['POST'])
def login():
    """User login"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'message': 'No data provided'}), 400
        
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({'message': 'Email and password are required'}), 400
        
        # Find user by email
        user = User.query.filter_by(email=email).first()
        
        if not user or not user.check_password(password):
            return jsonify({'message': 'Invalid email or password'}), 401
        
        # Generate token
        token = AuthUtils.generate_token(user.id)
        
        return jsonify({
            'message': 'Login successful',
            'user': user.to_dict(),
            'token': token
        }), 200
    
    except Exception as e:
        return jsonify({'message': f'Login error: {str(e)}'}), 500


@auth_bp.route('/profile', methods=['GET'])
def get_profile():
    """Get current user profile"""
    try: 
        token = None
        
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(" ")[1]
            except IndexError: 
                return jsonify({'message': 'Invalid token format'}), 401
        
        if not token: 
            return jsonify({'message':  'Token is missing'}), 401
        
        user = AuthUtils.get_current_user(token)
        if not user: 
            return jsonify({'message':  'Invalid or expired token'}), 401
        
        return jsonify({
            'user': user.to_dict()
        }), 200
    
    except Exception as e: 
        return jsonify({'message': f'Error:  {str(e)}'}), 500


@auth_bp.route('/logout', methods=['POST'])
def logout():
    """User logout (token blacklisting can be implemented if needed)"""
    try:
        # For now, logout is handled on the client side by removing the token
        return jsonify({'message': 'Logout successful'}), 200
    except Exception as e:
        return jsonify({'message': f'Logout error: {str(e)}'}), 500


@auth_bp.route('/refresh', methods=['POST'])
def refresh_token():
    """Refresh JWT token"""
    try:
        token = None
        
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
        
        # Generate new token
        new_token = AuthUtils.generate_token(user.id)
        
        return jsonify({
            'token': new_token,
            'user': user.to_dict()
        }), 200
    
    except Exception as e:
        return jsonify({'message': f'Error: {str(e)}'}), 500