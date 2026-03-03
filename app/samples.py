import numpy as np
from diagnostics import diagnose_single_chain, diagnose_multiple_chains, MultiChainDiagnostics

class Chain:
    def __init__(
            self, 
            betas: np.ndarray, 
            gammas: list[np.ndarray], 
            betas_burnin: np.ndarray,
            gammas_burnin: list[np.ndarray]
        ) -> None:
        self.betas = betas
        self.gammas = gammas
        self.betas_burnin = betas_burnin
        self.gammas_burnin = gammas_burnin

        self.length = betas.shape[0]
        self.burnin = betas_burnin.shape[0]
        self.nbetas = betas.shape[1]
        self.ngammas = len(gammas)

        self.beta_diagnostics = [diagnose_single_chain(betas[:, i]) for i in range(self.nbetas)]
        self.gamma_diagnostics = [[diagnose_single_chain(gamma[:, j]) for j in range(gamma.shape[1])] for gamma in gammas]
        self.burnin_beta_diagnostics = [diagnose_single_chain(betas_burnin[:, i]) for i in range(self.nbetas)]
        self.burnin_gamma_diagnostics = [[diagnose_single_chain(gamma[:, j]) for j in range(gamma.shape[1])] for gamma in gammas_burnin]



class SampleData:
    def __init__(self) -> None:
        self.nchains = 0
        self.chains = [] # type: list[Chain]
        self.beta_aggr_diagnostics = [] # type: list[MultiChainDiagnostics]
        self.gamma_aggr_diagnostics = [] # type: list[list[MultiChainDiagnostics]]
        self.filename = "na"
    
    def load_from_json(self, file_name) -> None:
        import json

        with open(file_name, 'r') as f:
            data = json.load(f)
        
        self.filename = file_name
        self.nchains = data['nchains']
        self.chains = []
        for chain in data['chains']:

            betas = np.array(chain['beta'])
            gammas = [np.array(gamma) for gamma in chain['gammas']]
            burnin_beta = np.array(chain['burninBeta'])
            burnin_gammas = [np.array(gamma) for gamma in chain['burninGammas']]

            self.chains.append(Chain(betas, gammas, burnin_beta, burnin_gammas))
        
        # Run aggregated diagnostics for each parameter.
        for i in range(self.chains[0].nbetas):
            beta_chains = [chain.betas[:, i] for chain in self.chains]
            self.beta_aggr_diagnostics.append(diagnose_multiple_chains(beta_chains))
        for i in range(self.chains[0].ngammas):
            gamma_diagnostic = []
            for j in range(self.chains[0].gammas[i].shape[1]):
                gamma_chains = [chain.gammas[i][:, j] for chain in self.chains]
                gamma_diagnostic.append(diagnose_multiple_chains(gamma_chains))
            self.gamma_aggr_diagnostics.append(gamma_diagnostic)
    
    def are_consistent(self):
        """
        Checks if all chains have the same number of samples and the same number of gammas.
        """
        for chain in self.chains:
            if chain.length != self.chains[0].length:
                print(f'Chain length mismatch: {chain.length} vs {self.chains[0].length}')
                return False

            if chain.betas.shape[0] != self.chains[0].betas.shape[0]:
                print(f'Beta shape mismatch: {chain.betas.shape} vs {self.chains[0].betas.shape}')
                return False

            if len(chain.gammas) != len(self.chains[0].gammas):
                print(f'Gamma length mismatch: {len(chain.gammas)} vs {len(self.chains[0].gammas)}')
                return False
            
        return True