class Solution:
    """
    Base class to represent the solution 

    Attributes:
        id (int): the identifier for the solution
        paper (str): Name of the reference paper 
        query (str): Research question or hypothesis 
        method (str): Method used for analysis 
        dataset_name (str): Name of the dataset
        answer (float): The estimated effect size 
        standard_error (float): Standard error of the estimate
        treat_var (str, optional): Name of the treatment variable, for regression discontinuity design 
                                 this may not always be applicable 
        outcome_var (str): Name of the outcome variable 
        control_vars (List[str], optional): list of the control variables / confounders 
        instrument_var (str): Name of the instrumental variable 
        running_var (str, optional): Name of the running variable, only used for regression discontinuity design 
        state_var (str, optional): Name of state variable, used in difference-in-differences / panel data analysis 
        time_var (str, optional): Name of the time variable, used in panel data analysis 
        interaction_var (str, optional): Name of the interaction variable that interacts with the treatment 
        is_rct (bool): Whether the study is from a randomized controlled trial 
        canonical_did (bool, optional): Whether the difference-in-differences model is the canonical 2 x 2 design
    """
    def __init__(self, id, paper, query, method, dataset_name, answer=None,
                 standard_error=None, treat_var=None, outcome_var=None,
                 control_vars=None, instrument_var=None, running_var=None, state_var=None,
                 time_var=None, interaction_var=None, is_rct=False, canonical_did=False,
                 is_multirct=False, did_term=None):

        ## the key attributes
        self.id = id
        self.paper = paper
        self.query = query
        self.method = method
        self.dataset_name = dataset_name 

        ## Numerical attributes 
        self.answer = answer 
        self.standard_error = standard_error

        ## Key attributes of the estimation model 
        self.treat_var = treat_var
        self.outcome_var = outcome_var
        self.control_vars = control_vars
        self.instrument_var = instrument_var
        self.running_var = running_var
        self.state_var = state_var
        self.time_var = time_var
        self.interaction_var = interaction_var
        self.is_rct = is_rct
        self.canonical_did = canonical_did
        self.is_multirct = is_multirct
        self.did_term = did_term

    def set_treat_var(self, treat_var):
        """
        Set the treatment variable

        Args:
            treat_var (str): Name of the treatment variable
        """

        self.treat_var = treat_var

    def set_outcome_var(self, outcome_var):
        """
        Set the outcome variable

        Args:
            outcome_var (str): Name of the outcome variable
        """

        self.outcome_var = outcome_var

    def set_control_vars(self, control_vars):
        """
        Set the control variables

        Args:
            control_vars (list): List of the control variables / confounders
        """

        self.control_vars = control_vars

    def set_instrument_var(self, instrument_var):
        """
        Set the instrumental variable

        Args:
            instrument_var (str): Name of the instrumental variable
        """

        self.instrument_var = instrument_var

    def set_running_var(self, running_var):
        """
        Set the running variable

        Args:
            running_var (str): Name of the running variable
        """

        self.running_var = running_var

    def set_state_var(self, state_var):
        """
        Set the state variable

        Args:
            state_var (str): Name of the state variable
        """

        self.state_var = state_var

    def set_time_var(self, time_var):
        """
        Set the time variable

        Args:
            time_var (str): Name of the time variable
        """

        self.time_var = time_var

    def set_interaction_var(self, interaction_var):
        """
        Set the interaction variable

        Args:
            interaction_var (str): Name of the interaction variable that interacts with the treatment
        """

        self.interaction_var = interaction_var

    def set_is_rct(self, is_rct):
        """
        Set an indicator for whether the study is from a randomized controlled trial or not

        Args:
            is_rct (bool): Whether the study is from a randomized controlled trial
        """
        self.is_rct = is_rct

    def set_answer(self, answer):
        """
        Set the estimated effect size

        Args:
            answer (float): The estimated effect size
        """
        self.answer = answer

    def set_standard_error(self, standard_error):
        """
        Set the standard error of the estimate

        Args:
            standard_error (float): The standard error of the estimate
        """
        self.standard_error = standard_error

    def to_row(self):
        """
        Converts the Solution to a dictionary row suitable for a CSV file.

        Returns:
            (dict): A dictionary mapping column names to values
        """
        return {
            "id": self.id,
            "paper_name": self.paper,
            "query": self.query,
            "dataset_name": self.dataset_name,
            "method": self.method,
            "answer": round(self.answer, 3) if self.answer is not None else None,
            "std_error": round(self.standard_error, 3) if self.standard_error is not None else None,
            "treatment": self.treat_var,
            "outcome": self.outcome_var,
            "controls": ",".join(self.control_vars) if self.control_vars else None,
            "instrument_var": self.instrument_var,
            "running_var": self.running_var,
            "temporal_var": self.time_var,
            "state_var": self.state_var,
            "interaction_var": self.interaction_var,
            "is_rct": int(self.is_rct),
            "is_multirct": int(self.is_multirct),
        }

    def __str__(self):

        base_rep = f"paper:{self.paper}\nquery:{self.query}"
        if self.answer is None:
            return base_rep
        else:
            return f"{base_rep}\nmethod:{self.method} answer: {self.answer:.3f}, std_error: {self.standard_error:.3f}\n"
        


class Paper:
    """
    Base class for representing the results used in the benchmark 

    Attributes:
        paper_id (str): The ID of the paper
        title (str): The title of the paper
        dataset_name (str): The name of the dataset associated with the paper 
        year (int): The year the paper was published
        domain (str): The domain of the paper
        n_solutions (int): The number of solutions in the paper
        solutions_dict (dict): A dictionary mapping query IDs to Solution objects corresponding 
                              to the solutions to the queries
    """

    def __init__(self, paper_id, title, dataset_name, year, domain, solutions_dict, 
                 is_multirct=False, is_rct=False, n_solutions=1):
        self.paper_id = paper_id
        self.title = title
        self.dataset_name = dataset_name
        self.year = year
        self.domain = domain
        self.solutions_dict = solutions_dict
        self.is_multirct = is_multirct
        self.is_rct = is_rct

    def __str__(self):

        return f"{self.title}"
    
    def set_solutions(self, solutions_dict):
        """
        Set the solutions for the paper

        Args:
            solutions_dict (dict): A dictionary mapping query IDs to Solution objects
        """
        self.solutions_dict = solutions_dict


    


    
