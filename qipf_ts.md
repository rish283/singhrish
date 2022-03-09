---
header-includes:
  - \usepackage{algorithm}
  - \usepackage{algpseudocode}
  - \usepackage{algorithmic}
  - \usepackage{algorithm2e}
---

<style TYPE="text/css">
code.has-jax {font: inherit; font-size: 100%; background: inherit; border: inherit;}
</style>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'] // removed 'code' entry
    }
});
MathJax.Hub.Queue(function() {
    var all = MathJax.Hub.getAllJax(), i;
    for(i = 0; i < all.length; i += 1) {
        all[i].SourceElement().parentNode.className += ' has-jax';
    }
});
</script>
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-AMS_HTML-full"></script>
  
<center> <h1> <ins>Time Series Data Analysis using Kernel Uncertainty Framework</ins> </h1> </center>
    
<figure>
<img style="float: center" src="qipf_tser.png" width="60%" height="60%">
<figcaption align = "center"><b>Decomposition of time series signal using QIPF uncertainty framework.</b></figcaption>
</figure>

<details>
  <summary> Abstract and Goal </summary>
we propose to utilize the QIPF framework which provides a completely data-adaptive and multi-moment uncertainty representation of a signal and is consequently able to quantify the local dynamics at each point in the sample space in an unsupervised manner with high sensitivity and specificity with repect ot the overall signal PDF. Through the use of the QIPF, we utilize concepts of quantum physics (which provides a principled quantification of particle-particle dynamics in a physical system) to interpret data. Consequently we introduce a new energy based information theoretic formulation to accomplish pattern recognition tasks associated with time series data that quantifies sample-by-sample dynamics of the signal (important in online time series analysis, which is not achievable by conventional methods).  We specifically explore applications like anomaly detection and clustering.
  </details>

<br />
<!-- ## Goal:
Our conjecture is that the best way to model non-stationary features of a time series signal is to use a dynamic embedding space that changes its local structure based on the evolution of the signal. To this end, we propose to utilize the QIPF uncertainty framework that, through its multiple uncertainty modes at each sample, is able to quantify local data dynamics relative to the signal's PDF in an unsupervised manner with high sensitivity and specificity. Through the use of the QIPF, we utilize concepts of quantum physics (which provides a principled quantification of particle-particle dynamics in a physical system) to interpret data. Consequently we introduce a new information theoretic framework to accomplish pattern recognition tasks associated with time series data that quantifies sample-by-sample dynamics of the signal (important in online time series analysi, which is not achievable by conventional methods. -->
<!-- <br /> -->
  
<!-- <br />
The problem is further made challenging by covariate shift of the test-set so that underlying distribution of input test data changes from $p(x|\lambda)$ during training to $p(x^*|\gamma)$ during testing (where $\lambda$ and $\gamma$ are parameters of the underlying distributions), while the target conditional distribution remains the same, i.e. $$p(y|x) = p(y^*|x^*)$$. -->
## Approach:
Our approach consists of three main steps:
 <br />
 <br />
$$\mathbf{1}$$. Estimation of PDF at a sample at time $$t$$ using information potential field (empirical estimate of kernel mean embedding): 
 <br />
 <br />
 $$p(x^t|x^0, x^1 ... x^{t-1}) \approx \psi_{\mathbf{x}}(x^t) = \frac{1}{n}\sum_{k=1}^{t-1}G_\sigma(x_k, x^t)$$.
    
---
    
$$\mathbf{2}$$. A Schrödinger's equation formulation over data PDF by assuming the IFP, $$\psi_{\mathbf{x}}(x^t)$$, to be a wave-function. This transforms the static PDF measure (the IPF) into a dynamic embedding that measures the local changes in the PDF at $$x^t$$: 
<br />
<br />
$$H_(x^t) = E_\mathbf{w}(x^t) + (\sigma^2/2)\frac{\nabla_y^2\psi_\mathbf{w}(x^t)}{\psi_\mathbf{w}(x^t)}$$ 
    
---
    
$$\mathbf{3}$$. Moment decomposition of $$H$$ to extract various uncertainty modes at $$x^t$$ which serve as dynamical features of the time-series at time t:
<br />
<br />
$$H^k(x^t) = E_\mathbf{w}^k(x^t) + (\sigma^2/2)\frac{\nabla_y^2\psi_\mathbf{w}^k(x^t)}{\psi_\mathbf{w}^k(x^t)}$$.
<br />
<br />
These stochastic features $$H^0(x^t), H^1(x^t), H^2(x^t) ...$$ are then utilized for applications like clustering or detection of change points in the time-series.

---
    
<figure>
<img style="float: center" src="/qspd7.jpg">
<figcaption align = "center"><b>Detailed depiction of approach: QIPF uncertainty decomposition of a time series.</b></figcaption>
</figure>

<br />

## Algorithm:
A pseudo-code for QIPF implementation is as follows:
<figure>
<img style="float: center" src="/alg.jpg" width="40%" height="40%">
</figure>
  
<br />
  
## Results:
  
<figure>
<img style="float: center" src="/tsr1.jpg">
<figcaption align = "center"><b>Analysis of mode locations of the sine wave in the space of data using different kernel widths. Solid colored lines represent the different QIPF modes. Dashed line represents the IPF.</b></figcaption>
</figure>
  
<br />
<br />
  
<figure>
<img style="float: center" src="/tsr2.jpg">
<figcaption align = "center"><b>Change point detection in time series: Last 1000 samples of drift datasets (top row), their corresponding QIPF mode standard deviations measured at each point (middle row) and corresponding the ROC curves (bottom row) for different methods measured in the range of 2000-3000 samples for both datasets. Black vertical lines (in the top row) mark the actual change points.</b></figcaption>
</figure>
  
<br />

## Related Papers:
  
Singh, R. and Principe, J., 2020, August. **Time Series Analysis using a Kernel based Multi-Modal Uncertainty Decomposition Framework**. In Conference on Uncertainty in Artificial Intelligence (pp. 1368-1377). PMLR. [(Paper Link)](http://proceedings.mlr.press/v124/singh20a.html)
<details>
<summary> Abstract </summary>
<br>
This paper proposes a kernel based information theoretic framework with quantum physical underpinnings for data characterization that is relevant to online time series applications such as unsupervised change point detection and whole sequence clustering. In this framework, we utilize the Gaussian kernel mean embedding metric for universal characterization of data PDF. We then utilize concepts of quantum physics to impart a local dynamical structure to characterized data PDF, resulting in a new energy based formulation. This facilitates a multi-modal physics based uncertainty representation of the signal PDF at each sample using Hermite polynomial projections. We demonstrate in this paper using synthesized datasets that such uncertainty features provide a better ability for online detection of statistical change points in time series data when compared to existing non-parametric and unsupervised methods. We also demonstrate a better ability of the framework in clustering time series sequences when compared to discrete wavelet transform features on a subset of VidTIMIT speaker recognition corpus.
</details>
