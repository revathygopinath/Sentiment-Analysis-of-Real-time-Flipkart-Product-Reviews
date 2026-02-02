#!/bin/bash

# Flipkart Sentiment Analysis - AWS EC2 Deployment Script
# This script automates the deployment of the Streamlit application on AWS EC2

echo "=========================================="
echo "Flipkart Sentiment Analysis Deployment"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
    print_error "Please do not run as root"
    exit 1
fi

# Step 1: Update system
print_info "Step 1: Updating system packages..."
sudo apt update && sudo apt upgrade -y
if [ $? -eq 0 ]; then
    print_success "System updated successfully"
else
    print_error "Failed to update system"
    exit 1
fi

# Step 2: Install Python and dependencies
print_info "Step 2: Installing Python and dependencies..."
sudo apt install -y python3 python3-pip python3-venv git
if [ $? -eq 0 ]; then
    print_success "Python and dependencies installed"
else
    print_error "Failed to install Python"
    exit 1
fi

# Step 3: Create project directory
print_info "Step 3: Setting up project directory..."
PROJECT_DIR="$HOME/flipkart_sentiment_analysis"
if [ -d "$PROJECT_DIR" ]; then
    print_info "Project directory already exists, updating..."
    cd "$PROJECT_DIR"
    git pull
else
    print_info "Cloning repository..."
    # Replace with your actual repository URL
    # git clone <your-repo-url> "$PROJECT_DIR"
    mkdir -p "$PROJECT_DIR"
    cd "$PROJECT_DIR"
fi
print_success "Project directory ready"

# Step 4: Create virtual environment
print_info "Step 4: Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_success "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Step 5: Install Python packages
print_info "Step 5: Installing Python packages..."
if [ -f "requirements.txt" ]; then
    pip install --upgrade pip
    pip install -r requirements.txt
    print_success "Python packages installed"
else
    print_error "requirements.txt not found"
    exit 1
fi

# Step 6: Download NLTK data
print_info "Step 6: Downloading NLTK data..."
python3 << EOF
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
print("NLTK data downloaded successfully")
EOF
print_success "NLTK data downloaded"

# Step 7: Check if models exist
print_info "Step 7: Checking for trained models..."
if [ ! -f "models/sentiment_model.pkl" ]; then
    print_info "Models not found. Training model..."
    python3 scripts/train_model.py
    if [ $? -eq 0 ]; then
        print_success "Model trained successfully"
    else
        print_error "Failed to train model"
        exit 1
    fi
else
    print_success "Models found"
fi

# Step 8: Install PM2 for process management (optional)
print_info "Step 8: Installing PM2 (optional)..."
if ! command -v pm2 &> /dev/null; then
    print_info "Installing Node.js and PM2..."
    sudo apt install -y nodejs npm
    sudo npm install -g pm2
    print_success "PM2 installed"
else
    print_success "PM2 already installed"
fi

# Step 9: Create startup script
print_info "Step 9: Creating startup script..."
cat > start_app.sh << 'EOFSTART'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0
EOFSTART

chmod +x start_app.sh
print_success "Startup script created"

# Step 10: Configure firewall
print_info "Step 10: Configuring firewall..."
sudo ufw allow 8501/tcp
sudo ufw allow 22/tcp
print_success "Firewall configured (port 8501 open)"

# Step 11: Display instructions
echo ""
echo "=========================================="
echo "Deployment Summary"
echo "=========================================="
print_success "Deployment completed successfully!"
echo ""
echo "Next steps:"
echo "1. Ensure AWS Security Group allows inbound traffic on port 8501"
echo "2. Start the application with one of these methods:"
echo ""
echo "   Method 1 - Direct start:"
echo "   $ ./start_app.sh"
echo ""
echo "   Method 2 - Using PM2 (recommended for production):"
echo "   $ pm2 start start_app.sh --name sentiment-app"
echo "   $ pm2 save"
echo "   $ pm2 startup"
echo ""
echo "3. Access your application at:"
echo "   http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8501"
echo ""
echo "4. To stop the application:"
echo "   - If using PM2: pm2 stop sentiment-app"
echo "   - If direct: Press Ctrl+C"
echo ""
echo "=========================================="

# Ask if user wants to start the app now
read -p "Do you want to start the application now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Starting application..."
    if command -v pm2 &> /dev/null; then
        pm2 start start_app.sh --name sentiment-app
        pm2 save
        print_success "Application started with PM2"
        echo "Run 'pm2 logs sentiment-app' to view logs"
    else
        print_info "Starting application directly..."
        ./start_app.sh
    fi
fi

echo ""
print_success "Setup complete! 🚀"
