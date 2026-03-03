import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from samples import SampleData


# Setup session variables -----------------------------------------------------

if 'sample_data' not in st.session_state:
    st.session_state.sample_data = None

if 'current_chain' not in st.session_state:
    st.session_state.current_chain = -1


# Build page -----------------------------------------------------------------
st.set_page_config(layout="wide")

st.title('Chain Diagnosis')

# Load data.
if st.sidebar.button('Load data'):
    data = SampleData()
    data.load_from_json('data/gibbs_march01 thinned.json')
    st.session_state.sample_data = data

# Display data.
if st.session_state.sample_data != None:
    sample_data = st.session_state.sample_data
    st.sidebar.write(f"Data: {sample_data.filename}")
    st.sidebar.write(f'Number of chains: {sample_data.nchains}')

    is_ok = sample_data.are_consistent()
    if is_ok:
        st.sidebar.success('All chains are consistent. ✅')
    else:
        st.sidebar.error('Chains are not consistent. ❌')

    # Select chain to display.
    st.session_state.current_chain = st.sidebar.selectbox('Select chain to display:', range(-1, sample_data.nchains), index=0)

# Display chain specific data.
if st.session_state.current_chain >= 0 and st.session_state.sample_data != None:
    sample_data = st.session_state.sample_data
    chain = sample_data.chains[st.session_state.current_chain]
    st.write(f'Chain {st.session_state.current_chain}:')
    st.write(f'Number of samples: {chain.length}')

    # Plot the distributions.
    st.write(f'## Distribution plots for chain {st.session_state.current_chain}')
    col1, col2 = st.columns(2)
    with col1: # Plot beta.
        st.write('Beta:')
        col11, col12 = st.columns(2)
        for i in range(chain.nbetas):
            # col = col11 if i % 2 == 0 else col12
            with col11 if i % 2 == 0 else col12:
                fig, ax = plt.subplots()
                ax.hist(chain.betas[:,i], bins=30, density=True)
                ax.set_title(f"Posterior of β[{i}]")
                ax.set_xlabel("Value")
                ax.set_ylabel("Density")
                ax.legend()
                st.pyplot(fig)

    with col2: # Plot gammas for each group.
        st.write('Gammas:')
        col21, col22 = st.columns(2)
        for i in range(chain.ngammas):
            with col21 if i % 2 == 0 else col22:
                fig, ax = plt.subplots()
                ax.hist(chain.gammas[i].flatten(), bins=30, density=True)
                ax.set_title(f"Posterior of γ group {i}")
                ax.set_xlabel("Value")
                ax.set_ylabel("Density")
                st.pyplot(fig)

   

    # Trace plots (burinin + sampling).
    st.write(f'## Trace plots for chain {st.session_state.current_chain}')
    col1, col2 = st.columns(2)
    with col1: # Plot beta.
        st.write('Beta:')
        fig, ax = plt.subplots()
        ax.plot(np.concatenate((chain.betas_burnin, chain.betas)), linewidth=0.8)
        ax.set_title(f"Trace (showing {chain.nbetas} β params)")
        ax.set_xlabel("Step")
        ax.axvline(chain.burnin, color='gray', linestyle='--', label='Burnin', linewidth=0.8)
        st.pyplot(fig)
    
    with col2: # Plot gammas for each group.
        st.write('Gammas:')
        col21, col22 = st.columns(2)
        for i in range(chain.ngammas):
            with col21 if i % 2 == 0 else col22:
                fig, ax = plt.subplots()
                ax.plot(np.concatenate((chain.gammas_burnin[i], chain.gammas[i])), linewidth=0.8)
                ax.set_title(f"Trace (showing {chain.gammas[i].shape[1]} γ params)")
                ax.set_xlabel("Step")
                ax.axvline(chain.burnin, color='gray', linestyle='--', label='Burnin', linewidth=0.8)
                st.pyplot(fig)

    # Show diagnostics.
    st.write(f'## Diagnostics for chain {st.session_state.current_chain}')
    beta_diagnostics = chain.beta_diagnostics
    # Display Geweke.
    col1, col2 = st.columns(2, border=True)
    with col1:
        st.write("#### Geweke for β")
        for i in range(chain.nbetas):
            z_score = abs(beta_diagnostics[i].geweke_z)
            z_ok = "✅"
            if z_score > 3:
                z_ok = "❌"
            elif z_score > 1.96:
                z_ok = "⚠️"
            st.write(f"β[{i}] Geweke z-score: {z_score:.2f} {z_ok}")
    with col2:
        for i in range(chain.ngammas):
            gamma_diagnostics = chain.gamma_diagnostics[i]
            st.write(f"#### Geweke for γ group {i}")
            for j in range(len(gamma_diagnostics)):
                z_score = abs(gamma_diagnostics[j].geweke_z)
                z_ok = "✅"
                if z_score > 3:
                    z_ok = "❌"
                elif z_score > 1.96:
                    z_ok = "⚠️"
                st.write(f"γ[{j}] Geweke z-score: {z_score:.2f} {z_ok}")


    # Display how many lags to reach < 0.1 autocorrelation.
    col1, col2 = st.columns(2, border=True)
    with col1:
        st.write("#### Autocorrelation Lag for β")
        for i in range(chain.nbetas):
            acf = beta_diagnostics[i].acf
            lag_01 = np.where(np.abs(acf) < 0.1)[0]
            lag_01 = lag_01[0] if len(lag_01) > 0 else len(acf)
            corr_ok = "✅"
            if lag_01 > 50:
                corr_ok = "❌"
            elif lag_01 > 20:
                corr_ok = "⚠️"
            st.write(f"β[{i}] autocorr < 0.1 at lag {lag_01} {corr_ok}")
    with col2:
        for i in range(chain.ngammas):
            gamma_diagnostics = chain.gamma_diagnostics[i]
            st.write(f"#### Autocorrelation Lag for γ group {i}")
            for j in range(len(gamma_diagnostics)):
                acf = gamma_diagnostics[j].acf
                lag_01 = np.where(np.abs(acf) < 0.1)[0]
                lag_01 = lag_01[0] if len(lag_01) > 0 else len(acf)
                corr_ok = "✅"
                if lag_01 > 50:
                    corr_ok = "❌"
                elif lag_01 > 20:
                    corr_ok = "⚠️"
                st.write(f"γ[{j}] autocorr < 0.1 at lag {lag_01} {corr_ok}")
    

    # Plot autocorrelation functions.
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Autocorrelation Function for β")
        col11, col12, col13 = st.columns(3)
        for i in range(chain.nbetas):
            acf = beta_diagnostics[i].acf
            with col11 if i % 3 == 0 else col12 if i % 3 == 1 else col13:
                fig, ax = plt.subplots()
                ax.bar(range(len(acf)), acf)
                ax.set_title(f"ACF of β[{i}]")
                ax.set_xlabel("Lag")
                ax.set_ylabel("Autocorrelation")
                st.pyplot(fig)
    with col2:
        for i in range(chain.ngammas):
            gamma_diagnostics = chain.gamma_diagnostics[i]
            st.write(f"#### Autocorrelation Function for γ group {i}")
            col21, col22, col23 = st.columns(3)
            for j in range(len(gamma_diagnostics)):
                acf = gamma_diagnostics[j].acf
                with col21 if j % 3 == 0 else col22 if j % 3 == 1 else col23:
                    fig, ax = plt.subplots()
                    ax.bar(range(len(acf)), acf)
                    ax.set_title(f"ACF of γ[{j}]")
                    ax.set_xlabel("Lag")
                    ax.set_ylabel("Autocorrelation")
                    st.pyplot(fig)
    
    # Display ess.
    col1, col2 = st.columns(2, border=True)
    with col1:
        st.write("#### Effective Sample Size for β")
        st.write("The **effective sample size (ESS)** tells you how many _independent_ " \
                 "samples the correlated MCMC draws are equivalent to. If the chain mixes" \
                 "perfectly, i.e. there is no autocorrelation, then ESS = N, but " \
                 "if the chain is highly autocorrelated, then ESS << N.")
        for i in range(chain.nbetas):
            ess = beta_diagnostics[i].ess
            ess_ok = "✅"
            if ess < 100:
                ess_ok = "❌"
            elif ess < 200:
                ess_ok = "⚠️"
            st.write(f"β[{i}] ESS: {ess:.1f} {ess_ok}")
    with col2:
        for i in range(chain.ngammas):
            gamma_diagnostics = chain.gamma_diagnostics[i]
            st.write(f"#### Effective Sample Size for γ group {i}")
            for j in range(len(gamma_diagnostics)):
                ess = gamma_diagnostics[j].ess
                ess_ok = "✅"
                if ess < 100:
                    ess_ok = "❌"
                elif ess < 200:
                    ess_ok = "⚠️"
                st.write(f"γ[{j}] ESS: {ess:.1f} {ess_ok}")
    
    # Display Monte Carlo standard error.
    col1, col2 = st.columns(2, border=True)
    with col1:
        st.write("#### Monte Carlo Standard Error for β")
        st.write("The MCSE tells us how much simulation noise remains in the estimate.")
        for i in range(chain.nbetas):
            mcse = beta_diagnostics[i].mcse
            mcse_ok = "✅"
            if mcse > 0.1:
                mcse_ok = "❌"
            elif mcse > 0.05:
                mcse_ok = "⚠️"
            st.write(f"β[{i}] MCSE: {mcse:.4f} {mcse_ok}")
    with col2:
        for i in range(chain.ngammas):
            gamma_diagnostics = chain.gamma_diagnostics[i]
            st.write(f"#### Monte Carlo Standard Error for γ group {i}")
            for j in range(len(gamma_diagnostics)):
                mcse = gamma_diagnostics[j].mcse
                mcse_ok = "✅"
                if mcse > 0.1:
                    mcse_ok = "❌"
                elif mcse > 0.05:
                    mcse_ok = "⚠️"
                st.write(f"γ[{j}] MCSE: {mcse:.4f} {mcse_ok}")


# Display aggregated diagnostics.
if st.session_state.sample_data != None:
    st.write("## Aggregated diagnostics across all chains")

    # Display Gelman-Rubin.
    col1, col2 = st.columns(2, border=True)
    with col1:
        st.write("#### Gelman-Rubin R-hat for β")
        for i in range(sample_data.chains[0].nbetas):
            rhat = sample_data.beta_aggr_diagnostics[i].gelman_rubin
            rhat_ok = "✅"
            if rhat > 1.05:
                rhat_ok = "❌"
            elif rhat > 1.01:
                rhat_ok = "⚠️"
            st.write(f"β[{i}] R-hat: {rhat:.3f} {rhat_ok}")
    with col2:
        for i in range(sample_data.chains[0].ngammas):
            gamma_diagnostic = sample_data.gamma_aggr_diagnostics[i]
            st.write(f"#### Gelman-Rubin R-hat for γ group {i}")
            for j in range(len(gamma_diagnostic)):
                rhat = gamma_diagnostic[j].gelman_rubin
                rhat_ok = "✅"
                if rhat > 1.05:
                    rhat_ok = "❌"
                elif rhat > 1.01:
                    rhat_ok = "⚠️"
                st.write(f"γ[{j}] R-hat: {rhat:.3f} {rhat_ok}")

    # Plot the cumulative Gelman-Rubin R-hat.
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Cumulative Gelman-Rubin R-hat for β")
        col11, col12 = st.columns(2)
        for i in range(sample_data.chains[0].nbetas):
            with col11 if i % 2 == 0 else col12:
                rhat_values = sample_data.beta_aggr_diagnostics[i].cum_gelman_rubin
                fig, ax = plt.subplots()
                ax.plot(rhat_values)
                ax.axhline(1.01, color='orange', linestyle='--', label='Warning threshold (1.01)')
                ax.axhline(1.05, color='red', linestyle='--', label='Failure threshold (1.05)')
                ax.set_title(f"Cumulative R-hat for β[{i}]")
                ax.set_xlabel("Number of samples")
                ax.set_ylabel("R-hat")
                ax.legend()
                st.pyplot(fig)
    with col2:
        for i in range(sample_data.chains[0].ngammas):
            gamma_diagnostic = sample_data.gamma_aggr_diagnostics[i]
            st.write(f"#### Cumulative Gelman-Rubin R-hat for γ group {i}")
            col21, col22 = st.columns(2)
            for j in range(len(gamma_diagnostic)):
                with col21 if j % 2 == 0 else col22:
                    rhat_values = gamma_diagnostic[j].cum_gelman_rubin
                    fig, ax = plt.subplots()
                    ax.plot(rhat_values)
                    ax.axhline(1.01, color='orange', linestyle='--', label='Warning threshold (1.01)')
                    ax.axhline(1.05, color='red', linestyle='--', label='Failure threshold (1.05)')
                    ax.set_title(f"Cumulative R-hat for γ[{j}]")
                    ax.set_xlabel("Number of samples")
                    ax.set_ylabel("R-hat")
                    ax.legend()
                    st.pyplot(fig)
