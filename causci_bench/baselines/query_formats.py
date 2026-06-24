import pandas as pd

def read_csv(path):
    """
    Reads a CSV file and returns a pandas DataFrame.
    Args:
        path (str): The path to the CSV file.
    Returns:
        pd.DataFrame: The DataFrame containing the CSV data.
    """

    try:
        df = pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding='latin1')
    return df

class QueryFormat:
    """A format of a query"""

    def __init__(self, query, dataset_path, dataset_description):
        self.query = query
        self.dataset_path = dataset_path
        self.dataset_description = dataset_description

    def get_query_format(self) -> str:
        pass

    def get_analysis_format(self, code_output: str) -> str:
        pass


class CausalQueryFormat(QueryFormat):
    def get_query_format(self, include_method_explanation=False):
        # Create a causal query based on the data and textual query
        df = read_csv(self.dataset_path)
        df_info = df.describe()
        columns_and_types = df.dtypes
        nan_per_column = df.isnull().sum(axis=0)
        if include_method_explanation:
            # Load prompt file relative to this module's directory
            from pathlib import Path
            prompt_path = Path(__file__).resolve().parent / "method_explanations.txt"
            with open(prompt_path) as file:
                method_explanation = file.read()
        else:
            method_explanation = ""

        query = f"""You are an expert in statistics and causal reasoning. You will answer a causal question on a tabular dataset.

The dataset is located at: {self.dataset_path}.

The dataset has the following description:
```
{self.dataset_description}
```

To help you understand it, here is the result of df.describe():
```
{df_info}
```

Here are the columns and their types:
```
{columns_and_types}
```

Here are the first 5 rows of the dataset:
```
{df.head()}
```

If there are fewer than 10 columns, here is the result of df.cov():
```
{(df.cov(numeric_only=True) if len(df.columns) < 10 else "Too many columns to compute covariance")}
```

Finally, here is the output of df.isnull().sum(axis=0):
```
{nan_per_column}
```

The causal question I would like you to answer is:
```
{self.query}
```

Here are some example methods; you can choose one from them: [
    IPW (Inverse Probability Weighting): Choose the right estimand (ATE/ATT/ATC), and compute the causal effect,
    Linear regression with control variables: Build a regression model with the treatment, outcome, and confounders/control variables, and compute the causal effects,
    Instrumental variable: Build an instrumental variable model, and compute the causal effects associated with the treatment variable,
    Matching: Choose the correct estimand (ATE/ATT/ATC), and match accordingly, and then compute the causal effects,
    Difference-in-differences: Build a difference-in-differences model, and output the coefficient of the variable of interest,
    Regression discontinuity design: Build a regression discontinuity design model, and output the coefficient of the variable of interest,
    Linear regression / difference-in-means: Either build a regression model consisting of the treatment and outcome variables, and compute the coefficient associated with the treatment variable or compute the difference in means across treatment and control groups,
    Generalized linear models / GLM: Build a GLM model, and output the coefficient of the variable of interest,
    Frontdoor adjustment: Build a causal graph, identify a mediator variable between the treatment and outcome, check for frontdoor criterion, and compute the causal effect using the frontdoor adjustment formula]

{method_explanation}

Using the descriptions and information from the dataset, write a Python code to build the causal inference model based on the method and variables you have selected, and computes the causal effect to answer the query. 
If you need to preprocess the data, please do so in the code. 
**Important: Only use these approved packages:**
- pandas (as pd)
- numpy (as np) 
- scipy
- scikit-learn (sklearn)
- statsmodels
- dowhy
- rdd (for regression discontinuity design)
- linearmodels 
- econml

Do not code yourself what is already implemented in the libraries. 
You need to print the final results, including:
    1. The causal effect (the value only)
    2. The standard deviation (the value only)
    3. The causal inference method that was used to compute the effect (the method name only)
    4. The treatment variable (the variable name only)
    5. The outcome variable (the variable name only)
    6. The mediator variable (the variable name only if frontdoor adjustment was used)
    7. RCT: True / False (NA if not sure; whether the data is from a randomized controlled trial or not)
    8. The covariates / control variables that were used in the causal inference model (the variable names only)
    9. Instrumental variable, if instrumental variable method was used (the variable name only)
    10. Running variable, if regression discontinuity design was used (the variable name only)
    11. Temporal variable, if difference-in-differences was used (the variable name only)
    12. Results of statistical tests, if applicable
    13. Brief Explanation for model choice
    14. The regression formula, if applicable.
If a variable is not applicable, print "NA" for it.

The code you output will be executed, and you will receive the output. Please make sure to output only one block of code, and make sure the code prints the result you are looking for at the end.
Everything between your first code block: '```python' and '```' will be executed. If there is an error, you will have several attempts to correct the code. 
Remember, the dataset is located at {self.dataset_path}.
"""
        return {"pre": [query]}

    def get_analysis_format(self, code_output: str) -> str:
        # Create a query for the analysis of the data
        query = f"""The code you provided has been executed, here is the output:
```
{code_output}
```
If the code returns an error, please provide a corrected version of the code. Output the entire code, not only the part that needs to be corrected.
Only provide the code if there is an error. Otherwise, if the previous code was executed, please provide a brief analysis of the results.
Use a single code block. If the code succeeds, do not add any new code, just provide the analysis.
"""
        return query


class CausalCoTFormat(CausalQueryFormat):
    def get_query_format(self, include_method_explanation=False):

        df = read_csv(self.dataset_path)
        columns_and_types = df.dtypes
        nan_per_column = df.isnull().sum(axis=0)
        if include_method_explanation:
            # Load prompt file relative to this module's directory
            from pathlib import Path
            prompt_path = Path(__file__).resolve().parent / "method_explanations.txt"
            with open(prompt_path) as file:
                method_explanation = file.read()
        else:
            method_explanation = ""

        query = f"""
You are an expert in causal inference. You will use a chain-of-thought approach to answer a causal question on a tabular dataset.
The dataset is located at : {self.dataset_path}
The dataset has the following description:
```
{self.dataset_description}
```
To help you understand it, here are the columns and their types:
``` 
{columns_and_types}
```
Here is the statistical summary of the dataset:
```
{df.describe()}
```
Here are the first 5 rows of the dataset:
```
{df.head()}
```
If there are fewer than 10 columns, here is the result of df.cov():
```
{(df.cov(numeric_only=True) if len(df.columns) < 10 else "Too many columns to compute covariance")}
```
Here is the output of df.isnull().sum(axis=0):
```
{nan_per_column}
```
The causal question I would like you to answer is:
```
{self.query}
```

Let us approach this problem step by step.  
Step 1. First, go through the dataset description and the columns and their types. Then, identify the treatment variable, the outcome variable, and the potential confounders.  
Explain your reasoning for choosing these variables. Remember, the dataset is located at: {self.dataset_path}.

Step 2. What would be the right estimand to consider for this problem? Then, choose the most appropriate method that can be used to estimate the causal effect. The available methods are:
Here are some example methods; you can choose one from them: [
    IPW (Inverse Probability Weighting): Choose the right estimand (ATE/ATT/ATC), and compute the causal effect,
    Linear regression with control variables: Build a regression model with the treatment, outcome, and confounders/control variables, and compute the causal effects,
    Instrumental variable: Build an instrumental variable model, and compute the causal effects associated with the treatment variable,
    Matching: Choose the correct estimand (ATE/ATT/ATC), and match accordingly, and then compute the causal effects,
    Difference-in-differences: Build a difference-in-differences model, and output the coefficient of the variable of interest,
    Regression discontinuity design: Build a regression discontinuity design model, and output the coefficient of the variable of interest,
    Linear regression / difference-in-means: Either build a regression model consisting of the treatment and outcome variables, and compute the coefficient associated with the treatment variable or compute the difference in means across treatment and control groups,
    Generalized linear models / GLM: Build a GLM model, and output the coefficient of the variable of interest,
    Frontdoor adjustment: Build a causal graph, identify a mediator variable between the treatment and outcome, check for frontdoor criterion, and compute the causal effect using the frontdoor adjustment formula
]

Explain why you chose the selected method, and how the data and its description support your choice. This means you should explain why the identification assumptions of the method are satisfied.  
{method_explanation}

Step 3. Next, we will plan the implementation. Before writing the code, describe your implementation process. This includes:  
1. Describing the necessary preprocessing steps.  
2. How we will select the variables to use in the model.  

Step 4. Finally, reflecting on the previous steps, write Python code to answer the causal question: {self.query}.  
Feel free to preprocess the data. 
**Important: Only use these approved packages:**
- pandas (as pd)
- numpy (as np) 
- scipy
- scikit-learn (sklearn)
- statsmodels
- dowhy
- rdd (for regression discontinuity design)
- linearmodels 
- econml
Use the methods from the above libraries to implement the method you chose. Be careful about implementation.  

Make sure the code prints the final results, including:  
You need to print the final results, including:
    1. The causal effect (the value only)
    2. The standard deviation (the value only)
    3. The causal inference method that was used to compute the effect (the method name only)
    4. RCT: True / False (NA is not sure; whether the data is from a randomized controlled trial or not)
    5. The treatment variable (the variable name only)
    6. The outcome variable (the variable name only)
    7. The mediator variable (the variable name only if frontdoor adjustment was used)
    8. The covariates / control variables that were used in the causal inference model (the variable names only)
    9. Instrumental variable, if instrumental variable method was used (the variable name only)
    10. Running variable, if regression discontinuity design was used (the variable name only)
    11. Temporal variable, if difference-in-differences was used (the variable name only)
    12. Results of statistical tests, if applicable
    13. Brief Explanation for model choice
    14. The regression formula, if applicable.
If a variable is not applicable, print "NA" for it.

The code you write will be executed, and you will next analyze the output. To ease the process, please output one block of code, and make sure the code prints the key results and values.  
Everything between your first code block: '```python' and '```' will be executed. If there is an error, you will have several attempts to correct the code. Hence, if there is an error, please fix it and re-run.
"""
        return {"pre": [query]}
    
    def get_analysis_format(self, code_output: str) -> str:

        query = f"""The code you provided has been executed, here is the output:
```
{code_output}
```
If the code returns an error, please provide a corrected version of the entire code. In this case, output exactly one Python code block containing the full corrected program.
Do not include any explanation outside the code.
If the code succeeds, do not output any code or code blocks. Instead, provide a plain-text analysis of the results. Ignore warnings.
"""
        return query 


class ReActFormat(QueryFormat):
    def get_query_format(self):
        # Create a ReAct query based on the data and textual query
        df = pd.read_csv(self.dataset_path)
        df_info = df.describe()
        columns_and_types = df.dtypes
        nan_per_column = df.isnull().sum(axis=0)
        format = f"""
Data Description:
{self.dataset_description}. The dataset is located at {self.dataset_path}.
You are a causal inference expert working with a pandas dataframe in Python. The name of the dataframe is `df`.
You should use the tools below to answer the causal question of interest:
`python_repl_ast`: A Python shell. Use this to execute Python commands. Input should be a valid Python command. When using this tool, sometimes output is abbreviated - make sure it does not look abbreviated before using it in your answer.

**Important: Only use these approved packages:**
- pandas (as pd)
- numpy (as np) 
- scipy
- scikit-learn (sklearn)
- statsmodels
- dowhy
- rdd (for regression discontinuity design)
- linearmodels 
- econml

### Use the following format:
Question: The input question you must answer
Thought: Your thoughts on what to do next. You need to think carefully.
Action: The action to take, should be python_repl_ast
Action Input: The input to the action, should be the code to execute
Observation: The result of the action
...(this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: The final answer to the original input question. Please provide a structured response including the following information. If a field is not applicable, use "NA".
- Method: [The method used]
- Causal Effect: [The causal effect estimate]
- Standard Error: [The standard error of the causal effect]
- Treatment Variable: [The treatment variable]
- Outcome Variable: [The outcome variable]
- Mediator Variable: [The mediator variable, if frontdoor adjustment was used, NA otherwise]
- RCT: [True / False indicating if the data is from a randomized controlled trial, NA if not sure]
- Covariates: [List of covariates and confounders used in the estimation model]
- Additional Variable: [Instrument, running variable, or temporal variable, if applicable]
- Results of Statistical Tests: [Key statistical results, if applicable]
- Explanation for Model Choice: [Explanation, if applicable]
- Regression Formula: [The regression formula, if applicable]

Note: Only import from the approved package list above. Do not use any other packages.
DO NOT create any plotting.
For all outputs in code, THE `print()` function MUST be called.
If you use Action in this step, stop after generating the Action Input and await the execution outcome from `python_repl_ast`.
If you output the Final Answer in this step, do not use Action.

Here is an example of using the `python_repl_ast`:
Action: python_repl_ast
Action Input:
```python
# Your code goes here - only use approved packages
import pandas as pd
import numpy as np
print(df.head())
```
Begin!
Question: {self.query}
Available causal inference methods:
    IPW (Inverse Probability Weighting): Choose the right estimand (ATE/ATT/ATC), and compute the causal effect
    Linear regression with control variables: Build a regression model with the treatment, outcome, and confounders/control variables, and compute the causal effects
    Instrumental variable: Build an instrumental variable model, and compute the causal effects associated with the treatment variable
    Matching: Choose the correct estimand (ATE/ATT/ATC), and match accordingly, and then compute the causal effects
    Difference-in-differences: Build a difference-in-differences model, and output the coefficient of the variable of interest
    Regression discontinuity design: Build a regression discontinuity design model, and output the coefficient of the variable of interest
    Linear regression / difference-in-means: Either build a regression model consisting of the treatment and outcome variables, and compute the coefficient associated with the treatment variable or compute the difference in means across treatment and control groups
    Generalized linear models / GLM: Build a GLM model, and output the coefficient of the variable of interest
    Frontdoor adjustment: Build a causal graph, identify a mediator variable between the treatment and outcome, check for frontdoor criterion, and compute the causal effect using the frontdoor adjustment formula

"""

        return {"pre": [format]}
    
    def get_analysis_format(self, code_output: str) -> str:
        # Create a query for the analysis of the data
        query = f"""The code you provided has been executed, here is the output:
```
{code_output}
```
"""
        return query

    
class ProgramOfThoughtsFormat(QueryFormat):
    def get_query_format(self):
        # Create a program of thoughts query based on the data and textual query
        df = pd.read_csv(self.dataset_path)
        df_info = df.describe()
        columns_and_types = df.dtypes
        nan_per_column = df.isnull().sum(axis=0)
        format = f"""
You are a causal inference expert. Your goal is to generate a causality-driven answer to the user query: "{self.query}" using the provided data. 
The description and the query can be found below. Please analyze the input information and write Python code that performs causal effect estimation. 
You can use the following libraries: pandas, numpy, scipy, sklearn, statsmodels, dowhy, rdd, linearmodels, and econml. 
The format of the code should be:
```python
def causal_analysis():
    # import libraries 
    # load data
    # identify treatment, outcome, confounders, control variables (pre-treatment variables)
    # select appropriate causal method, and method-specific variables 
    # estimate causal effect and standard error
    # print results (12 items listed below). This is important
    # return a dictionary containing the 12 items listed below 

result = causal_analysis() 
```
Available causal inference methods: IPW (Inverse Probability Weighting), Linear regression with control variables, 
Instrumental variable, Matching, Difference-in-Differences, Regression Discontinuity Design, 
Linear Regression/Difference-in-Means, Generalized linear models, Frontdoor adjustment. 

Print the following 12 items in the code:
1. Causal effect  
2. Standard error 
 3. Method name  
4. RCT (True/False/NA)  
5. Treatment variable  
6. Outcome variable  
7. Mediator variable  
8. Covariates used  
9. Additional variable  
10. Statistical test results  
11. Model choice explanation  
12. Regression formula (if applicable)
Print "NA" for non-applicable fields.

Here is information about the data. 

Data Description: {self.dataset_description}
Dataset Location: {self.dataset_path}
Columns and types: {columns_and_types}
First 10 rows: {df.head(10)}
Missing values: {nan_per_column}
Likewise, the query is: {self.query}
"""
        return {"pre": [format]}

    def get_analysis_format(self, code_output: str) -> str:
        # Create a query for the analysis of the data
        query = f"""The code you provided has been executed, here is the output:
```
{code_output}
```
Can you please provide an analysis of the results? Keep the analysis concise and focus on the key findings.
If the code returns an error, please provide a corrected version of the code. Output the entire code, not only the part that needs to be corrected.
Only provide the code if there is an error. Otherwise, if the previous code was executed, please provide a brief analysis of the results.
Use a single code block. If the code succeeds, do not add any new code, just provide the analysis.
"""
        return query

