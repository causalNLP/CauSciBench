import pandas as pd
import re
import json

from .chatbot import Chatbot
from .query_formats import QueryFormat, CausalQueryFormat
from .coderunner import CodeRunner

from typing import Optional


def print_color(text, color):
    """Print text in color"""

    print(f"\033[{color}m{text}\033[0m")


def find_code(reply, language="python"):
    """Find python code in a string `reply`"""

    code_start = reply.find("```{}".format(language))
    if code_start == -1:
        return None
    code_start += len("```python")
    code_end = reply[code_start:].find("```")
    if code_end == -1:
        return None
    code_end += code_start
    code = reply[code_start:code_end]
    return code


class Baseline:
    """A conversational chatbot that has access to a dataset"""

    def __init__(self, chatbot: Chatbot, safe_exec=True, persistent=False, 
                 session_timeout=3600, max_retries=3, worker_id=None) -> None:
        self.chatbot = chatbot
        self.code_runner = CodeRunner(safe_exec=safe_exec, persistent=persistent, 
                                      session_timeout=session_timeout, worker_id=worker_id)
        self.max_retries = max_retries
        self.persistent = persistent

    def get_final_result(self):

        """Get the final result from the chatbot in a structured JSON format."""

        prompt = """
Please provide a final summary of the analysis in a single, well-formed JSON object. The JSON object should have the following keys. If a field is not applicable, use `null`.

- `method`: The name of the primary causal inference method used (e.g., "Propensity Score Weighting", "Difference-in-Differences", "Frontdoor Estimation").
- `causal_effect`: The estimated causal effect. Provide this as a numerical value.
- `standard_deviation`: The standard deviation of the causal effect estimate, if available.
- `treatment_variable`: The name of the treatment variable.
- `rct`: Boolean indicating if the data is from a randomized controlled trial (true/false), or `null` if unsure.
- `outcome_variable`: The name of the outcome variable.
- `mediators`: The name of the mediator variable, if applicable. 
- `covariates`: A list of control / pre-treatment variables (for regression based estimators) or confounders used in causal inference process.
- `instrument_variable`: The name of the instrumental variable, if applicable.
- `running_variable`: The name of the running variable for Regression Discontinuity, if applicable.
- `temporal_variable`: The name of the time variable for Difference-in-Differences, if applicable.
- `statistical_test_results`: A summary of key statistical test results, like p-values or confidence intervals.
- `explanation_for_model_choice`: A brief explanation for why the chosen causal method was appropriate for this analysis.
- `regression_equation`: The exact regression equation if a regression model was used.

Output the JSON object only, without any additional text or explanation. Ensure the JSON is properly formatted and valid.
"""
        json_reply = self.chatbot.ask(prompt)

        try:
            # Remove any markdown code blocks if present
            if "```json" in json_reply:
                json_start = json_reply.find("```json") + 7
                json_end = json_reply.find("```", json_start)
                if json_end != -1:
                    json_reply = json_reply[json_start:json_end].strip()
            elif "```" in json_reply:
                json_start = json_reply.find("```") + 3
                json_end = json_reply.find("```", json_start)
                if json_end != -1:
                    json_reply = json_reply[json_start:json_end].strip()
            
            # Remove any comments (// style)
            lines = json_reply.split('\n')
            cleaned_lines = []
            for line in lines:
                if '//' in line:
                    line = line[:line.index('//')]
                cleaned_lines.append(line)
            json_reply = '\n'.join(cleaned_lines)
            
            # Find the start and end of the JSON object
            json_start = json_reply.find('{')
            json_end = json_reply.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                json_str = json_reply[json_start:json_end]
                return json.loads(json_str)
            else:
                return {"error": "Could not find a valid JSON object in the response.", 
                        "raw_response": json_reply}
        except json.JSONDecodeError as e:
            return {"error": f"Failed to decode JSON: {str(e)}", 
                    "raw_response": json_reply}

    def get_variable_value(self, variable_name):
        """Get the value of a variable in the persistent environment."""

        if not self.persistent:
            return "Error: Not in persistent mode"
        
        return self.code_runner.get_variable_value(variable_name)

    def get_defined_variables(self):
        """Get a list of all defined variables in the environment."""

        if not self.persistent:
            return "Error: Not in persistent mode"
        
        return self.code_runner.get_defined_variables()

    def start_persistent_session(self):
        """Start a persistent Python session."""

        if not self.persistent:
            return "Error: Not in persistent mode"
        
        return self.code_runner.start_persistent_container()

    def stop_persistent_session(self):
        """Stop the persistent Python session."""

        if not self.persistent:
            return "Error: Not in persistent mode"
        
        return self.code_runner.stop_persistent_container()

    def is_session_active(self):
        """Check if the persistent session is active."""

        if not self.persistent:
            return False
        
        return self.code_runner.is_container_running()

    def upload_file(self, local_path, container_path=None):
        """Upload a file from the local machine to the container."""

        if not self.persistent:
            return "Error: Not in persistent mode"
        
        return self.code_runner.upload_file(local_path, container_path)

    def download_file(self, container_path, local_path=None):
        """Download a file from the container to the local machine."""

        if not self.persistent:
            return "Error: Not in persistent mode"
        
        return self.code_runner.download_file(container_path, local_path)

    def list_files(self, directory='.'):
        """List files in a directory in the container."""

        if not self.persistent:
            return "Error: Not in persistent mode"
        
        return self.code_runner.list_files(directory)

    def answer(self, query, dataset_path, dataset_description="", qf=CausalQueryFormat, post_steps=False):
        """Answer a causal query using the dataset path (a df)"""
        
        self.chatbot.delete_history()
        
        # Initialize the query format
        query_format = qf(query, dataset_path, dataset_description)
        
        # Get the query format
        queries = query_format.get_query_format()
        
        reply = ""
        
        # Handle pre-analysis queries
        if "pre" in queries:
            for q in queries["pre"]:
                # Pre-analysis queries
                print_color(q, 32)
                reply = self.chatbot.ask(q)
                print_color(reply, 33)
        
        codes = []
        code_outputs = []
        
        # Run code while the model outputs code
        for retry_count in range(self.max_retries):
            code = find_code(reply)
            if code is None:
                break

            codes.append(code)

            code_output = self.code_runner.run_code(code)
            code_outputs.append(code_output)

            # Ask the chatbot to analyze the results
            analysis_query = query_format.get_analysis_format(code_output)
            print_color(analysis_query, 32)
            reply = self.chatbot.ask(analysis_query)
            print_color(reply, 33)

        # Post-analysis queries
        if post_steps and "post" in queries:
            for q in queries["post"]:
                print_color(q, 32)
                reply = self.chatbot.ask(q)
                print_color(reply, 33)

        final_result = self.get_final_result()
        chat_history = self.chatbot.conversation_history
        
        # Return the results
        return {
            "query": query,
            "codes": codes,
            "code_outputs": code_outputs,
            "chat_history": chat_history,
            "retries": max(0, len(codes) - 1),  
            "final_result": final_result,}