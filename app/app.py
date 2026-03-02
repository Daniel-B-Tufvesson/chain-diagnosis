import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from data.samples import SampleData


# Setup session variables -----------------------------------------------------

if 'sample_data' not in st.session_state:
    st.session_state.sample_data = None

if 'current_chain' not in st.session_state:
    st.session_state.current_chain = -1


# Build page -----------------------------------------------------------------
st.set_page_config(layout="wide")

st.title('Chain Diagnosis')

# Load data.
if st.button('Load data'):
    data = SampleData()
    data.load_from_json('data/gibbs_march01.json')
    st.session_state.sample_data = data

# Display data.
if st.session_state.sample_data != None:
    sample_data = st.session_state.sample_data
    st.write(f'Number of chains: {sample_data.nchains}')

    is_ok = sample_data.are_consistent()
    if is_ok:
        st.success('All chains are consistent. ✅')
    else:
        st.error('Chains are not consistent. ❌')

    # Select chain to display.
    st.session_state.current_chain = st.selectbox('Select chain to display:', range(-1, sample_data.nchains), index=0)

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

   

    # Trace plots.
    st.write(f'## Trace plots for chain {st.session_state.current_chain}')
    col1, col2 = st.columns(2)
    with col1: # Plot beta.
        st.write('Beta:')
        fig, ax = plt.subplots()
        ax.plot(chain.betas, linewidth=0.8)
        ax.set_title(f"Trace (showing {chain.nbetas} β params)")
        ax.set_xlabel("Step")
        st.pyplot(fig)
    
    with col2: # Plot gammas for each group.
        st.write('Gammas:')
        col21, col22 = st.columns(2)
        for i in range(chain.ngammas):
            with col21 if i % 2 == 0 else col22:
                fig, ax = plt.subplots()
                ax.plot(chain.gammas[i], linewidth=0.8)
                ax.set_title(f"Trace (showing {chain.gammas[i].shape[1]} γ params)")
                ax.set_xlabel("Step")
                st.pyplot(fig)


