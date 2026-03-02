import numpy as np
from diagnostics import diagnose_single_chain

class Chain:
    def __init__(self, betas: np.ndarray, gammas: list[np.ndarray]) -> None:
        self.betas = betas
        self.gammas = gammas
        self.length = betas.shape[0]
        self.nbetas = betas.shape[1]
        self.ngammas = len(gammas)
        self.beta_diagnostics = [diagnose_single_chain(betas[:, i]) for i in range(self.nbetas)]
        self.gamma_diagnostics = [[diagnose_single_chain(gamma[:, j]) for j in range(gamma.shape[1])] for gamma in gammas]


class SampleData:
    def __init__(self) -> None:
        self.nchains = 0
        self.chains = [] # type: list[Chain]
    
    def load_from_json(self, file_name) -> None:
        import json

        with open(file_name, 'r') as f:
            data = json.load(f)
        
        self.nchains = data['nchains'][0]
        self.chains = []
        for chain in data['chains']:
            betas = np.array(chain['beta'])
            gammas = [np.array(gamma) for gamma in chain['gammas']]
            self.chains.append(Chain(betas, gammas))
    
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