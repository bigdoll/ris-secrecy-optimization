#!/bin/bash

# Set the path to your project directory
PROJECT_DIR="/Users/robertkukufotock/Documents/METAWIRELESS/Research-Codes/spawc2024"
cd "$PROJECT_DIR" || { echo "Directory not found!"; exit 1; }

# Check if Git is initialized
if [ ! -d ".git" ]; then
    echo "Git not initialized. Initializing Git repository..."
    git init
    echo "Git repository initialized."
fi

# Check if the GitHub remote is set up
REMOTE_URL=$(git remote get-url origin 2>/dev/null)
if [ -z "$REMOTE_URL" ]; then
    echo "No remote found. Setting up remote repository..."
    read -p "Enter your GitHub repository URL (https://github.com/username/repo.git): " REPO_URL
    git remote add origin "$REPO_URL"
    echo "Remote repository set to $REPO_URL."
fi

# Check if the main branch exists, if not create it
if ! git rev-parse --verify main >/dev/null 2>&1; then
    echo "Main branch not found. Creating main branch..."
    git checkout -b main
fi

# Fetch the latest changes from the remote repository to avoid conflicts
echo "Fetching the latest changes from the remote repository..."
git fetch origin

# Attempt to merge the remote changes, with error handling
echo "Merging remote changes..."
if ! git merge origin/main --allow-unrelated-histories --no-edit; then
    echo "Merge failed. Aborting the merge and reverting to a clean state."
    git merge --abort
    exit 1
fi

# Check for changes
if [[ $(git status --porcelain) ]]; then
    echo "Changes detected. Adding changes to Git..."

    # Add all changes (tracked files, new files, and deletions)
    git add --all

    # Get the current date and time for commit message
    COMMIT_MESSAGE="Automated commit: $(date +'%Y-%m-%d %H:%M:%S')"

    # Commit the changes
    git commit -m "$COMMIT_MESSAGE"
    echo "Changes committed."

    # Push changes to GitHub
    git push origin main
    echo "Changes pushed to GitHub!"
else
    echo "No changes detected. Nothing to commit."
fi


# # Set the path to your project directory
# PROJECT_DIR="/Users/robertkukufotock/Documents/METAWIRELESS/Research-Codes/spawc2024"
# cd "$PROJECT_DIR" || { echo "Directory not found!"; exit 1; }

# # Check if Git is initialized
# if [ ! -d ".git" ]; then
#     echo "Git not initialized. Initializing Git repository..."
#     git init
#     echo "Git repository initialized."
# fi

# # Check if the GitHub remote is set up
# REMOTE_URL=$(git remote get-url origin 2>/dev/null)
# if [ -z "$REMOTE_URL" ]; then
#     echo "No remote found. Setting up remote repository..."
#     read -p "Enter your GitHub repository URL (https://github.com/username/repo.git): " REPO_URL
#     git remote add origin "$REPO_URL"
#     echo "Remote repository set to $REPO_URL."
# fi

# # Check if the main branch exists, if not create it
# if ! git rev-parse --verify main >/dev/null 2>&1; then
#     echo "Main branch not found. Creating main branch..."
#     git checkout -b main
# fi

# # Fetch the latest changes from the remote repository to avoid conflicts
# echo "Fetching the latest changes from the remote repository..."
# git fetch origin

# # Try to merge the remote changes if any
# echo "Merging remote changes..."
# git merge origin/main --no-edit

# # Check for changes
# if [[ $(git status --porcelain) ]]; then
#     echo "Changes detected. Adding changes to Git..."

#     # Add all changes (tracked files, new files, and deletions)
#     git add --all

#     # Get the current date and time for commit message
#     COMMIT_MESSAGE="Automated commit: $(date +'%Y-%m-%d %H:%M:%S')"

#     # Commit the changes
#     git commit -m "$COMMIT_MESSAGE"
#     echo "Changes committed."

#     # Push changes to GitHub
#     git push origin main
#     echo "Changes pushed to GitHub!"
# else
#     echo "No changes detected. Nothing to commit."
# fi
