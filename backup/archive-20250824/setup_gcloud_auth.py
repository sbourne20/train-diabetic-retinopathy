#!/usr/bin/env python3
"""
Setup Google Cloud authentication for the project
"""

import subprocess
import sys
import os
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_gcloud_installed() -> bool:
    """Check if gcloud CLI is installed."""
    try:
        result = subprocess.run(['gcloud', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"gcloud CLI found: {result.stdout.split()[0]}")
            return True
    except FileNotFoundError:
        pass
    
    logger.error("gcloud CLI not found!")
    return False

def check_authentication() -> bool:
    """Check if user is authenticated."""
    try:
        result = subprocess.run(['gcloud', 'auth', 'list', '--filter=status:ACTIVE', 
                               '--format=value(account)'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout.strip():
            accounts = result.stdout.strip().split('\n')
            logger.info(f"Authenticated accounts: {accounts}")
            return True
        else:
            logger.warning("No active authentication found")
            return False
            
    except Exception as e:
        logger.error(f"Error checking authentication: {e}")
        return False

def check_application_default_credentials() -> bool:
    """Check if Application Default Credentials are set up."""
    try:
        import google.auth
        credentials, project_id = google.auth.default()
        logger.info(f"Application Default Credentials found for project: {project_id}")
        return True
    except Exception as e:
        logger.warning(f"Application Default Credentials not found: {e}")
        return False

def setup_authentication(project_id: Optional[str] = None) -> bool:
    """Set up Google Cloud authentication."""
    
    logger.info("Setting up Google Cloud authentication...")
    
    # Check if already authenticated
    if check_authentication() and check_application_default_credentials():
        logger.info("✅ Already authenticated!")
        return True
    
    # Login to Google Cloud
    if not check_authentication():
        logger.info("🔐 Please log in to Google Cloud...")
        try:
            subprocess.run(['gcloud', 'auth', 'login'], check=True)
            logger.info("✅ Successfully logged in to Google Cloud")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to log in: {e}")
            return False
    
    # Set up Application Default Credentials
    if not check_application_default_credentials():
        logger.info("🔑 Setting up Application Default Credentials...")
        try:
            subprocess.run(['gcloud', 'auth', 'application-default', 'login'], 
                         check=True)
            logger.info("✅ Application Default Credentials set up successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to set up Application Default Credentials: {e}")
            return False
    
    # Set project if provided
    if project_id:
        try:
            subprocess.run(['gcloud', 'config', 'set', 'project', project_id], 
                         check=True)
            logger.info(f"✅ Project set to: {project_id}")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to set project: {e}")
            return False
    
    return True

def enable_required_apis(project_id: str) -> bool:
    """Enable required Google Cloud APIs."""
    
    apis = [
        'storage.googleapis.com',
        'aiplatform.googleapis.com', 
        'compute.googleapis.com'
    ]
    
    logger.info("🔌 Enabling required APIs...")
    
    for api in apis:
        try:
            logger.info(f"Enabling {api}...")
            subprocess.run(['gcloud', 'services', 'enable', api], 
                         check=True, capture_output=True)
            logger.info(f"✅ {api} enabled")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to enable {api}: {e}")
            return False
    
    return True

def verify_setup(project_id: str) -> bool:
    """Verify the complete setup."""
    
    logger.info("🔍 Verifying setup...")
    
    # Check authentication
    if not check_authentication():
        logger.error("❌ Authentication failed")
        return False
    
    # Check Application Default Credentials
    if not check_application_default_credentials():
        logger.error("❌ Application Default Credentials not working")
        return False
    
    # Test Google Cloud Storage access
    try:
        from google.cloud import storage
        client = storage.Client(project=project_id)
        # Try to list buckets (this will fail if no permission, but won't crash)
        try:
            list(client.list_buckets(max_results=1))
            logger.info("✅ Google Cloud Storage access verified")
        except Exception as e:
            logger.warning(f"⚠️  Limited storage access: {e}")
            logger.info("This might be normal if you don't have storage permissions yet")
            
    except Exception as e:
        logger.error(f"❌ Google Cloud Storage test failed: {e}")
        return False
    
    logger.info("✅ Setup verification completed!")
    return True

def get_installation_instructions():
    """Get installation instructions for gcloud CLI."""
    
    import platform
    system = platform.system().lower()
    
    instructions = {
        'darwin': """
🍎 macOS Installation:
1. Using Homebrew (recommended):
   brew install --cask google-cloud-sdk

2. Direct download:
   curl https://sdk.cloud.google.com | bash
   exec -l $SHELL  # Restart shell
        """,
        'linux': """
🐧 Linux Installation:
1. Ubuntu/Debian:
   sudo apt-get update
   sudo apt-get install apt-transport-https ca-certificates gnupg
   echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
   curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
   sudo apt-get update && sudo apt-get install google-cloud-cli

2. Direct download:
   curl https://sdk.cloud.google.com | bash
   exec -l $SHELL  # Restart shell
        """,
        'windows': """
🪟 Windows Installation:
1. Download installer from:
   https://cloud.google.com/sdk/docs/install

2. Using Chocolatey:
   choco install gcloudsdk

3. Using pip:
   pip install gcloud
        """
    }
    
    return instructions.get(system, instructions['linux'])

def main():
    """Main function."""
    
    print("🔐 Google Cloud Authentication Setup")
    print("=" * 40)
    
    # Check if gcloud is installed
    if not check_gcloud_installed():
        print("❌ gcloud CLI is not installed!")
        print(get_installation_instructions())
        print("\nAfter installation, run this script again.")
        return False
    
    # Get project ID
    project_id = input("Enter your Google Cloud Project ID: ").strip()
    if not project_id:
        print("❌ Project ID is required!")
        return False
    
    # Setup authentication
    if not setup_authentication(project_id):
        print("❌ Authentication setup failed!")
        return False
    
    # Enable APIs
    if not enable_required_apis(project_id):
        print("⚠️  Some APIs might not be enabled. You can enable them manually.")
    
    # Verify setup
    if verify_setup(project_id):
        print("\n🎉 Authentication setup completed successfully!")
        print(f"Project: {project_id}")
        print("\nYou can now run:")
        print(f"python setup_gcs.py --project_id {project_id} --bucket_name your-bucket-name")
        return True
    else:
        print("❌ Setup verification failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)