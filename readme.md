​# Execution Instructions
​
# Python version 3.8.10
​
To create a virtual environment and install requirements in Python 3.8.10 on different operating systems, follow the instructions below:
​
### For Windows:
​
Open the Command Prompt by pressing Win + R, typing "cmd", and pressing Enter.
​

Change the directory to the desired location for your project:
​
`cd C:\path\to\project`

Create a new virtual environment using the venv module:

`python -m venv myenv`

Activate the virtual environment:

`myenv\Scripts\activate`

Install the project requirements using pip:
​
`pip install -r requirements.txt`


### For Linux/Mac:
​
Open a terminal.

Change the directory to the desired location for your project:

`cd /path/to/project`
​
Create a new virtual environment using the venv module:
​
`python3 -m venv myenv`
​

Activate the virtual environment:
​
`source myenv/bin/activate`
​

Install the project requirements using pip:
​
`pip install -r requirements.txt`
​

These instructions assume you have Python 3.8.10 installed and added to your system's PATH variable.
​

Then you can run the notebook or use the command to run the app:

`streamlit run llm_app.py`