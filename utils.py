import pandas  as pd
import scipy.stats as stats
import numpy as np 
from hmmlearn import hmm
import plotly.graph_objects as go
import os.path
import plotly.express as px
def fetch_fi2010() -> pd.DataFrame:
    """
    Load the FI2010 dataset with no auction.
    Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data with Machine
    Learning Methods. A Ntakaris, M Magris, J Kanniainen, M Gabbouj, A Iosifidis.
    arXiv:1705.03233 [cs.CE].  https://arxiv.org/abs/1705.03233
    Parameters
    """
    url = "https://raw.githubusercontent.com/simaki/fi2010/main/data/data.csv"
    if 'df_fi2010' in locals():
        return locals()["df_fi2010"]
    elif os.path.isfile('fi2010.parquet.gzip'):
        return pd.read_parquet("fi2010.parquet.gzip")
    else:
        df_fi2010 = pd.read_csv(url, index_col=0)
        df_fi2010.to_parquet("fi2010.parquet.gzip",compression='gzip')
    return df_fi2010



def distribution_stats(returns):
    print('Stylised fact of the retrun distribution')
    print('-' * 50)
    print('Length of the return series:', returns.shape)
    print('Mean:', returns.mean()*100, '%')
    print('Standard Deviation:', returns.std()*100, '%')
    print('Skew:', returns.skew())
    print('Kurtosis:', returns.kurtosis())


def test_distribution(returns, mode, alpha = 0.05):
    # Input: return and test('mean' or 'normal'), alpha in decimals

    if (mode == 'mean'):
        t_stat, p = stats.ttest_1samp(returns, popmean=0, alternative='two-sided')
    elif (mode == 'normal'):
        k2, p = stats.normaltest(returns)
    
    def test(p, alpha):
        print("p = {:g}".format(p))
        if p < alpha: 
            print("The null hypothesis can be rejected")
        else:
            print("The null hypothesis cannot be rejected")
            
    return test(p, alpha)   



def fit_HMM(data, n_states):
    model = hmm.GaussianHMM(n_components=n_states, n_iter=1000).fit(data)
    
    relabeled_states = model.predict(data)
    post_prob = np.array(model.predict_proba(data))

    # fit HMM parameters
    mus = np.squeeze(model.means_)
    sigmas = np.squeeze(np.sqrt(model.covars_))
    transmat = np.array(model.transmat_)

    return relabeled_states, mus, sigmas, transmat, post_prob, model



def fit_best_HMM(train, test, n_states, iterations=1000):
    train_vals = np.expand_dims(train, 1)
    train_vals = np.reshape(train_vals,[len(train),1])

    test_vals = np.expand_dims(test, 1)
    test_vals = np.reshape(test_vals,[len(test),1])

    best_score=None
    for index in range(iterations):
        relabeled_states, mus, sigmas, transmat, post_prob, model = fit_HMM(train_vals, n_states)
        score = model.score(test_vals)
        if best_score is None or score > best_score:
            best_model = (relabeled_states, mus, sigmas, transmat, post_prob, model)
            best_score = score

    relabeled_states, mus, sigmas, transmat, post_prob, model = best_model
    return relabeled_states, mus, sigmas, transmat, post_prob, model




def plot_states_shades(data, df_post_prob):
    fig = go.Figure()
    dates= list(range(len(data)))

    for state in range(df_post_prob.shape[1]):
        fig.add_trace(go.Scatter(x=dates, y=df_post_prob.iloc[:,state], name = state, mode='lines', line_shape='hv',
                                line=dict(width=0.5, color=px.colors.qualitative.Plotly[state]), 
                                stackgroup='two', yaxis = 'y2'))
    fig.add_trace(go.Scatter(x=dates, y=data, name="DATA", mode='lines', line_shape='hv', yaxis = 'y1'))
                            
    # Show plot 
    fig.update_layout(
        title = ("Volatility Regime - HMM"),
        yaxis=dict(title="R"),
        yaxis2=dict(title="Posterier Probability", overlaying="y1", side="right"),
        legend = dict(orientation = 'h')
    )
    fig.show()


def plot_states(data, relabeled_states):
    fig = go.Figure()
    dates= list(range(len(relabeled_states)))
    fig.add_trace(go.Scatter(x=dates, y=relabeled_states, name = 'State', mode='lines'))
    fig.add_trace(go.Scatter(x=dates, y=data, name = "Data", mode='lines', stackgroup='two', yaxis = 'y2'))
    # Show plot 
    fig.update_layout(
        title = ("Volatility Regime - HMM"),
        yaxis=dict(title="State"),
        yaxis2=dict(title="Absolute Returns", overlaying="y1", side="right",tickformat= ',.0%'),
        legend = dict(orientation = 'h')
    )
    fig.show()


def detect_DC(data, threshold=0.0005):
    P_EXT=[]
    for index in range(len(data)):
        current_price= data[index]
        if index==0:
            P_EXT.append([index,current_price])
        else:
            if abs((current_price - P_EXT[-1][1])/P_EXT[-1][1])>=threshold :
                P_EXT.append([index,current_price])
    TMV_EXT = [[curr[0],(curr[1]-prev[1])/(prev[1]*threshold)] for prev, curr in zip(P_EXT[:-1], P_EXT[1:])] 
    T= [[curr[0],curr[0]-prev[0]] for prev, curr in zip(P_EXT[:-1], P_EXT[1:])] 
    R=[[tmv[0], threshold*tmv[1]/t[1]] for tmv, t in zip(TMV_EXT, T)]
    R= list(map(lambda el: el[1],R))
    return R