# Contributing to PyTheranostics

The following guidelines will help developers of PyTheranostics work quickly and collaboratively.

## Running the code

### 1. Download and Install the Code and Requirements

- Install Python version 3.8 or higher.
- Install Git.
- Open a Bash terminal and navigate to where you want to put the PyTheranostics source code.
- Clone the repository.
  ```bash
  git clone https://github.com/qurit/PyTheranostics.git
  ```
- Navigate to the root `PyTheranostics` directory.
  ```bash
  cd PyTheranostics
  ```
- Install the packages.
  ```bash
  pip install --upgrade pip
  install -r requirements.txt
  ```

### 2. Run an Example Dosimetry Calculation

// todo

### 3. Run Tests

// todo

## Reporting Issues

- Use [GitHub Issues](https://github.com/qurit/PyTheranostics/issues) to report bugs or request features.
- Provide as much detail as possible, including steps to reproduce bugs and your environment.
- **Do not assign the issue** to anyone except yourself - Assigned issues are in-progress issues.
- Discuss new issues with the development team.

## Contributing

### 1. Pick Up an Issue

- Browse open (unassigned) issues and find one you would like to work on, or report a new issue.
- **Self-assign the issue** by clicking the “Assignees” section on the issue page. This lets others know you are working on it, even if you are working locally.
- If you have questions or need clarification, comment on the issue before starting.

### 2. Work on the Issue

- Create a new branch from `dev` with a name of the format `<issue#>-<descriptive-name>` e.g.:
  ```bash
  git switch dev    # always branch from dev
  git pull          # make sure your dev branch is up to date
  git switch -c 42-calculate-voxel-density   # -c specifies a new branch
  ```
- Make your changes, maintaining a consistent code style.
- Write or update tests as needed.  //todo
- Ensure all tests pass locally.  //todo
- At least once per day, push your changes to maintain backups.
  ```bash
  git push -u origin HEAD  # first time
  # OR
  git push  # subsequent times
  ```

### 3. Submit a Pull Request (PR)

#### Synchronize your branch

- Merge the latest dev changes into your code.
  ```bash
  git merge origin/dev
  ```
- Check that the merge did not break anything.
- Push your branch.
  ```bash
  git push
  ```
- In Github, open a pull request **into the `dev` branch**.
- In the PR description, reference the issue using `closes #<issue-number>` (e.g., `closes #42`). This will automatically close the issue when the PR is merged.
- Provide a clear description of your changes and any relevant context.
- Describe how to run the code that you have added or changed.
- Assign at least one person to review the code.  If you are not sure who to add, add Carlos (the repository owner).

### 4. Code Review

- Wait for your PR to be reviewed.
- Address any requested changes by continuing development in [section 2](#2-work-on-the-issue).  When you push your changes, the PR will be automatically updated.
- Once approved, click the `merge` button.  This will automatically close the PR and the issue that it was associated with.
## Example Workflow

1. **Find and self-assign an issue:**  
    Browse open (unassigned) issues or create a new one. On the issue page, assign yourself using the “Assignees” section.

2. **Create a branch from `dev`:**  
    ```bash
    git switch dev
    git pull
    git switch -c 42-fix-readme-typo  # Replace 42 and name as appropriate
    ```

3. **Make your changes:**  
    - Edit the relevant files to address the issue. Maintain consistent code style.

4. **Write or update tests (if needed):**  
    - Add or update tests to cover your changes. Ensure all tests pass locally.

5. **Commit and push your changes:**  
    ```bash
    git add README.md
    git commit -m "Fix typo in README"
    git push -u origin HEAD
    ```

6. **Synchronize with latest `dev` changes:**  
    ```bash
    git merge origin/dev
    git push
    ```

7. **Open a Pull Request into `dev`:**  
    - In GitHub, open a PR targeting the `dev` branch.
    - In the PR description, reference the issue (e.g., `closes #42`).
    - Provide a clear description of your changes and how to test them.
    - Assign at least one reviewer (add Carlos if unsure).

8. **Respond to code review:**  
    - Address any feedback by pushing additional commits to your branch.

9. **Merge the PR:**  
    - Once approved, click the `merge` button. The PR and associated issue will be closed automatically.

---

Thank you for helping make PyTheranostics better!
