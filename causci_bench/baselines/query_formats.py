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


class DirectFormat(QueryFormat):
    """
    Base class for representing direct prompting for causal inference. 
    The prompting structure should not capture the causal inference pipeline, but asks the LLM in a more straightforward manner
    to build a causal inference model and implement it on the dataset to answer a causal question. 
    """
    def get_query_format(self, include_method_explanation=False):
        # Create a causal query based on the data and textual query
        if include_method_explanation:
            # Load prompt file relative to this module's directory
            from pathlib import Path
            prompt_path = Path(__file__).resolve().parent / "method_explanations.txt"
            with open(prompt_path) as file:
                method_explanation = file.read()
        else:
            method_explanation = ""

        query = f"""
You are an expert in causal inference. Your goal is to develop a causal inference model and implement it on the dataset to answer a causal question.

The dataset is located at: {self.dataset_path}.

The dataset has the following description:
```
{self.dataset_description}
```

The causal question I would like you to answer is:
```
{self.query}
```
You can choose one of the following methods:
   IPW (Inverse Probability Weighting) with an appropriate estimand (ATE/ATT/ATC), Linear regression with control variables, Instrumental Variable,
   Matching with an appropriate estimand (ATE/ATT/ATC), Difference-in-Differences, Regression Discontinuity Design,
   Difference-in-means (equivalent to linear regression with outcome and treatment), Generalized linear models / GLMs, Frontdoor adjustment.
You must justify your choice of method based on the data and its description. You can also perform statistical tests to support your choice.

Then, write a Python code to implement the method you have selected. Make sure to print the key steps and results.

**Important: Only use these approved packages:** pandas, numpy, scipy, scikit-learn (sklearn), statsmodels, dowhy,
              rdd (for regression discontinuity design), linearmodels, econml

You need to print the following:
    1. Effect: The causal effect (the value only)
    2. Standard Deviation: The standard deviation (the value only)
    3. Method: The causal inference method that was used
    4. Justification: Justification for the method choice i.e. how the data and its description justify the identification assumptions. 
    5. Treatment: The treatment variable (the variable name only)
    6. Outcome: The outcome variable (the variable name only)
    7. Mediator: The mediator variable (the variable name only if frontdoor adjustment was used)
    8. RCT: True / False (NA if not sure; whether the data is from a randomized controlled trial or not)
    9. Confounders / Controls: The confounders / control variables that were used in the causal inference model (the variable names only)
    10. Instrument: The instrument, if instrumental variable method was used (the variable name only)
    11. Running Variable: The running variable, if regression discontinuity design was used (the variable name only)
    12. Temporal Variable: The temporal variable, if difference-in-differences was used (the variable name only)
    13. Statistical results: The key statistical results, if applicable.
    14. Formula: The regression formula, if applicable.
    15. Interpretation: The final interpretation of the result, with respect to the causal question of interest.

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


class CausalCoTFormat(QueryFormat):
    """
    Base class for representing chain-of-thought prompting for causal inference. 
    We essentially encode the causal inference pipeline in the prompt. 
    """
    def get_query_format(self, include_method_explanation=False):

        if include_method_explanation:
            # Load prompt file relative to this module's directory
            from pathlib import Path
            prompt_path = Path(__file__).resolve().parent / "method_explanations.txt"
            with open(prompt_path) as file:
                method_explanation = file.read()
        else:
            method_explanation = ""

        query = f"""
You are an expert in causal inference. Your goal is to develop a causal inference model and implement it on the dataset to answer a causal question.

The dataset is located at: {self.dataset_path}.

The dataset has the following description:
```
{self.dataset_description}
```

The causal question I would like you to answer is:
```
{self.query}
```

Let us approach this problem step by step.
Step 1. First, go through the dataset description and the query. Then, identify the treatment variable and the outcome variable from the dataset.
        Also, reason why these variables would be appropriate.
Step 2. Next, reason about the potential confounders affecting both treatment and outcome i.e. given the setting, what variables could
        affect both treatment and outcome, and why?
Step 3. What would be the appropriate estimand to consider for this problem?
        Then, we will be choosing a suitable inference method to estimate the estimand. You can choose from the following methods:
           IPW (Inverse Probability Weighting), Linear regression with control variables, Instrumental Variable,
           Matching with an appropriate estimand, Difference-in-Differences, Regression Discontinuity Design,
           Difference-in-means (equivalent to linear regression with outcome and treatment), Generalized linear models / GLMs, Frontdoor adjustment.
        Carefully reason about the identification assumptions of each of the above methods, how they relate to the data and its description, and whether they
        are satisfied or not.
        Based on your reasoning, select the most appropriate method to estimate the effect. Justify why the selected method is appropriate, and 
        can plausibly identify the causal effect. You can either argue qualitatively and/or perform statistical tests to support your choice.

Step 4. Next, we will write the Python code to implement the method you have selected. In doing so, carefully think about the key pre-processing steps.
        Make sure to print the key steps and the causal effect estimate with its standard deviation. 
**Important: Only use these approved packages:** pandas, numpy, scipy, scikit-learn (sklearn), statsmodels, dowhy,
             rdd (for regression discontinuity design), linearmodels, econml

You need to print the following:
    1. Effect: The causal effect (the value only)
    2. Standard Deviation: The standard deviation (the value only)
    3. Method: The causal inference method that was used
    4. Justification: Justification for the method choice i.e. how does the data and its description justify the identification assumptions.
       You can do this by providing an explanation or interpreting the result statistics / diagnostic test.
    5. Treatment: The treatment variable (the variable name only)
    6. Outcome: The outcome variable (the variable name only)
    7. Mediator: The mediator variable (the variable name only if frontdoor adjustment was used)
    8. RCT: True / False (NA if not sure; whether the data is from a randomized controlled trial or not)
    9. Confounders / Controls: The confounders / control variables that were used in the causal inference model (the variable names only)
    10. Instrument: The instrument, if instrumental variable method was used (the variable name only)
    11. Running Variable: The running variable, if regression discontinuity design was used (the variable name only)
    12. Temporal Variable: The temporal variable, if difference-in-differences was used (the variable name only)
    13. Statistical results: The key statistical results, if applicable.
    14. Formula: The regression formula, if applicable.
    15. Interpretation: The final interpretation of the result, with respect to the causal question of interest.

If a variable is not applicable, print "NA" for it.

The code you write will be executed, and you will next analyze the output. To ease the process, please output one block of code, and make sure the code prints the key results and values.
Everything between your first code block: '```python' and '```' will be executed. If there is an error, you will have several attempts to correct the code.

Remember, the dataset is located at {self.dataset_path}.
"""
        return {"pre": [query]}
    
    def get_analysis_format(self, code_output: str) -> str:

        query = f"""The code you provided has been executed, here is the output:
```
{code_output}
```
If the code returns an error, please provide a corrected version of the code. Output the entire code, not only the part that needs to be corrected.
Only provide the code if there is an error. Otherwise, if the previous code was executed, please provide a brief analysis of the results.
Use a single code block. If the code succeeds, do not add any new code, just provide the analysis.
"""
        return query 


class ReActFormat(QueryFormat):
    """
    Base class for representing ReAct-based prompting for causal inference. 
    """
    def get_query_format(self):
        # Create a ReAct query based on the data and textual query
        format = f"""
You are an expert in causal inference. Your goal is to develop a causal inference model and implement it on the dataset to answer a causal question.

The dataset is located at: {self.dataset_path}.

The dataset has the following description:

    {self.dataset_description}

The causal question I would like you to answer is:

    {self.query}

You may use the following tool:

    python_repl_ast: A Python shell used to execute Python code. The input must be valid Python code.

Important: Only use these approved python packages: pandas, numpy, scipy, scikit-learn (sklearn), statsmodels, dowhy, 
           rdd (for regression discontinuity design), linearmodels, econml

Additional constraints:
    - Every code block must use print() to output results / findings of interest.
    - When using python_repl_ast, stop after producing the Action Input and wait for the Observation before continuing.
    - Always wrap Action Input code in ```python ... ``` (not ```python_repl_ast or any other variant).

For reference, here is a typical causal inference pipeline. 
1.  Explore the dataset to understand its structure, data types, missing values, and other characteristics that might be helpful.
2.  Identify the treatment and outcome variables from the dataset
3.  Identify potential confounders affecting both treatment and outcome.
4.  Select the most appropriate causal inference method from the list below: 
       IPW (Inverse Probability Weighting), Linear regression with control variables, Instrumental Variable, Matching with an appropriate estimand, 
       Difference-in-Differences, Regression Discontinuity Design, Difference-in-means (equivalent to linear regression with outcome and treatment), 
       Generalized linear models / GLMs, Frontdoor adjustment
5.  Justify why the method is appropriate for the given scenario i.e. how the data and its description justify the identification assumptions, 
    and the can plausibly identify the causal effect. 
    You can either argue qualitatively and/or perform statistical tests to support your choice.
6.  Implement the method in Python using the dataframe df.
7.  Compute the causal effect and standard error.

Use the following format for reasoning and action:
    Question: The causal question you must answer
    Thought: Your thoughts about what to do next.
    Action: The action you need to take; it should be python_repl_ast
    Action Input: The input to the action i.e. the code to execute. Every code block must use print() to output results / findings of interest. 
    Observation: Observation and Interpretation of the output from action input. 
    The Thought -> Action -> Action Input -> Observation steps can repeat multiple times until you determine the answer.
    Thought: I now know the final answer. 
    Final Answer: The final answer that includes,
        1.  Effect: The causal effect (the value only)
        2.  Standard Deviation: The standard deviation (the value only)
        3.  Method: The causal inference method that was used
        4.  Justification: Justification for the method choice i.e. how the data and its description justify the identification assumptions. 
        5.  Treatment: The treatment variable (the variable name only)
        6.  Outcome: The outcome variable (the variable name only)
        7.  Mediator: The mediator variable (the variable name only if frontdoor adjustment was used)
        8.  RCT: True / False (NA if not sure; whether the data is from a randomized controlled trial or not)
        9.  Confounders / Controls: The confounders / control variables that were used in the causal inference model (the variable names only)
        10. Instrument: The instrument, if instrumental variable method was used (the variable name only)
        11. Running Variable: The running variable, if regression discontinuity design was used (the variable name only)
        12. Temporal Variable: The temporal variable, if difference-in-differences was used (the variable name only)
        13. Statistical Results: The key statistical results, if applicable
        14. Formula: The regression formula, if applicable
        15. Interpretation: The final interpretation of the result with respect to the causal question of interest
    
If a field is not applicable, return “NA”.

Here is an example of using the python_repl_ast:
Action: python_repl_ast
Action Input:
```python
# Your code goes here - only use approved packages
import pandas as pd
import numpy as np
print(df.head())
```
Begin!
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
    """
    Base class for representing program-of-thoughts prompting for causal inference. 
    While the initial steps such as identifying treatment, outcome, and confounders are the same, the difference lies in how 
    one selects the methods. Here, we mostly perform diagnostic tests to assess the assumptions, and then select the appropriate method. 
    """
    def get_query_format(self):
        # Create a program of thoughts query based on the data and textual query
        format = f"""
You are an expert in causal inference. Your goal is to write a Python program to implement a causal inference model on the provided dataset to answer a query of interest.

The dataset is located at: {self.dataset_path}

The dataset has the following description:

{self.dataset_description}

The causal question to answer is:

{self.query}

You need to select and implement one of the following methods: 
  IPW (Inverse Probability Weighting), Linear regression with control variables, Instrumental Variable, Matching with an appropriate estimand, 
  Difference-in-Differences, Regression Discontinuity Design, Difference-in-means (equivalent to linear regression with outcome and treatment), 
  Generalized linear models / GLMs, Frontdoor adjustment

Important: Only use these approved packages: pandas, numpy, scipy, scikit-learn (sklearn), statsmodels, dowhy, 
rdd (for regression discontinuity design), linearmodels, econml

Write the solution as a Python program that performs these steps.

Use the following structure:

# Step 1: Load dataset 
# Step 2: Exploratory analysis of the data 
# Step 3: Identify treatment and outcome variables from the dataset
# Step 4: Identify the confounders. 
# Step 5: Perform diagnostic tests to assess the assumptions associated with the candidate methods given above. 
# Step 6: Select the most appropriate method based on the diagnostic tests and qualitatively reasoning about the assumptions in the context of the data and its description.
# Step 7: Build the final causal model 
# Step 8: Display the final result. 

Make sure to print results for each of the above steps. 

The code you write will be executed, and you will next analyze the output. To ease the process, please output one block of code, and make sure the code prints the key results and values.
Everything between your first code block: '```python' and '```' will be executed. If there is an error, you will have several attempts to correct the code.

The final outputs must include:
    1.  Effect: The causal effect (the value only)
    2.  Standard Deviation: The standard deviation or standard error of the causal effect estimate (the value only)
    3.  Method: The causal inference method that was used
    4.  Justification: Justification for the method choice, i.e. how the data and its description support the identification assumptions. 
    5.  Treatment: The treatment variable (the variable name only)
    6.  Outcome: The outcome variable (the variable name only)
    7.  Mediator: The mediator variable (the variable name only if frontdoor adjustment was used)
    8.  RCT: True / False (NA if not sure; whether the data is from a randomized controlled trial or not)
    9.  Confounders / Controls: The confounders / control variables that were used in the causal inference model (the variable names only)
    10. Instrument: The instrument, if the instrumental variable method was used (the variable name only)
    11. Running Variable: The running variable, if regression discontinuity design was used (the variable name only)
    12. Temporal Variable: The temporal variable, if difference-in-differences was used (the variable name only)
    13. Statistical Results: The key statistical results, if applicable
    14. Formula: The regression formula, if applicable
    15. Interpretation: The final interpretation of the result with respect to the causal question of interest

If a field is not applicable, return “NA”.
Remember, the output must be a single Python program. 
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


class ChainReactFormat(QueryFormat):
    """
    Base class for representing a prompting format that combines CoT for reasoning and ReAct for implementation. 
    We use CoT to reason about the identification assumptions, and then use ReAct to implement the relevant diagnostic tests and the causal inference model.
    """
    def get_query_format(self):
        format = f"""
You are an expert in causal inference. Your goal is to develop a causal inference model and implement it on the dataset to answer a causal question.
We will work in two phases. First, reason rigorously about the model, and then implement it in ReAct format.

The dataset is located at: {self.dataset_path}

The dataset has the following description:

{self.dataset_description}

The causal question is:

{self.query}

Reasoning Phase:

1.  Recall the dataset description and the query. Based on this information, identify the treatment variable and the outcome variable in the dataset to answer the query.
2.  Given the scenario, identify the possible confounders, i.e., the variables affecting both the treatment and the outcome. Reason why these variables act as confounders.
3.  Consider the following inference methods:
    IPW (Inverse Probability Weighting), Linear regression with control variables, Instrumental Variable, Matching with an appropriate estimand,
    Difference-in-Differences, Regression Discontinuity Design, Difference-in-means (equivalent to linear regression with outcome and treatment), 
    Generalized linear models / GLMs, Frontdoor adjustment.
    Carefully think about the assumptions underlying each of the above methods. For each method, reason whether its identification assumptions
    are plausible given the dataset description and the causal query i.e. whether the method can plausibly identify the causal effect or not.
4.  After evaluating all candidate methods and their assumptions, select the most appropriate method (only 1) for estimating the causal effect. 
    Clearly state the assumptions of the selection method and justify why they are plausible given the dataset description and the causal query. 

Next, we will implement the method based on what you have reasoned above.

Implementation Phase (ReAct):
You are working with a pandas dataframe in Python named df. You may use the following tool:
    python_repl_ast: A Python shell used to execute Python code. The input must be valid Python code.

Important: Only use these approved python packages: pandas, numpy, scipy, scikit-learn (sklearn), statsmodels, dowhy, rdd (for regression discontinuity design), 
linearmodels, econml.

Use code execution to:
1.  Explore the dataset.
2.  Evaluate the assumptions of the selected method. Altogether, assess whether the selected method is appropriate or not in this scenario.
3.  Revise the method if the diagnostics + qualitative assessment do not support it i.e. select an alternative method and justify its assumptions.
4.  Implement the final causal model. 
5.  Display the final output. 

Important:
    - Every code block must use print() to output results and findings of interest.
    - Always wrap Action Input code in ```python ... ``` (not ```python_repl_ast or any other variant).

Here is an example of using the python_repl_ast:
Action: python_repl_ast
Action Input:
```python
# Your code goes here - only use approved packages
import pandas as pd
import numpy as np
df = pd.read_csv("{self.dataset_path}")
print(df.head())
```

Interaction Format:
    Question: The causal question you must answer.
    Reasoning Phase: The output from the reasoning phase.
    Thought: What action you need to take next.
    Action: The action you need to take; it should be python_repl_ast.
    Action Input: The code to execute, wrapped in ```python ... ```. Always use print() to display the results.
    Observation: The output returned from the action. Do not write additional code here.
    
    The Thought -> Action -> Action Input -> Observation steps may repeat until you determine the final answer.

When you have reached the final answer, output:
Thought: I now know the final answer.
Final Answer:

1.  Effect: The causal effect (value only)
2.  Standard Deviation: The standard deviation (value only)
3.  Method: The causal inference method that was used
4.  Justification: Explanation of why the method is appropriate and how the data supports the identification assumptions
5.  Treatment: The treatment variable (variable name only)
6.  Outcome: The outcome variable (variable name only)
7.  Mediator: The mediator variable (variable name only if frontdoor adjustment was used)
8.  RCT: True / False (NA if not sure)
9.  Confounders / Controls: The confounders / control variables used in the causal model (variable names only)
10. Instrument: The instrument, if instrumental variable method was used (variable name only)
11. Running Variable: The running variable, if regression discontinuity design was used (variable name only)
12. Temporal Variable: The temporal variable, if difference-in-differences was used (variable name only)
13. Statistical Results: The key statistical results, if applicable
14. Formula: The regression formula, if applicable
15. Interpretation: The final interpretation of the result with respect to the causal question

If a field is not applicable, return “NA”.

Begin!
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


