import numpy as np
class GaussianTargetEncoder():
        
    def __init__(self, group_cols, target_col="target", prior_cols=None):
        self.group_cols = group_cols
        self.target_col = target_col
        self.prior_cols = prior_cols

    def _get_prior(self, df):
        if self.prior_cols is None:
            prior = np.full(len(df), df[self.target_col].mean())
        else:
            prior = df[self.prior_cols].mean(1)
        return prior
                    
    def fit(self, df):
        self.stats = df.assign(mu_prior=self._get_prior(df), y=df[self.target_col])
        self.stats = self.stats.groupby(self.group_cols).agg(
            n        = ("y", "count"),
            mu_mle   = ("y", np.mean),
            sig2_mle = ("y", np.var),
            mu_prior = ("mu_prior", np.mean),
        )        
    
    def transform(self, df, prior_precision=1000, stat_type="mean"):
        
        precision = prior_precision + self.stats.n/self.stats.sig2_mle
        
        if stat_type == "mean":
            numer = prior_precision*self.stats.mu_prior\
                    + self.stats.n/self.stats.sig2_mle*self.stats.mu_mle
            denom = precision
        elif stat_type == "var":
            numer = 1.0
            denom = precision
        elif stat_type == "precision":
            numer = precision
            denom = 1.0
        else: 
            raise ValueError(f"stat_type={stat_type} not recognized.")
        
        mapper = dict(zip(self.stats.index, numer / denom))
        if isinstance(self.group_cols, str):
            keys = df[self.group_cols].values.tolist()
        elif len(self.group_cols) == 1:
            keys = df[self.group_cols[0]].values.tolist()
        else:
            keys = zip(*[df[x] for x in self.group_cols])
        
        values = np.array([mapper.get(k) for k in keys]).astype(float)
        
        prior = self._get_prior(df)
        values[~np.isfinite(values)] = prior[~np.isfinite(values)]
        
        return values
    
    def fit_transform(self, df, *args, **kwargs):
        self.fit(df)
        return self.transform(df, *args, **kwargs)