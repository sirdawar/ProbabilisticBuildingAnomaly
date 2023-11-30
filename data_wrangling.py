def add_lagged_values(l_ds, params = { 'group_by': None, 'arrange_by': None, 'value_col': None, 'shift': [1], 'col_namespace': 'lag_%sh'}, inplace = True ):
    group_by = params['group_by']
    arrange_by = params['arrange_by']
    value_col = params['value_col']
    new_col_namespace = params['col_namespace']

    assert value_col is not None, 'The value_col parameter need to be specified!'
    assert new_col_namespace is not None and (new_col_namespace % str(1)) != new_col_namespace, 'The col_namespace parameter need to be specified and contain \%\s special character!'
    
    if inplace == True:
        l_ds_copy = l_ds
    else:
        l_ds_copy = l_ds.copy()

    if arrange_by is not None:
        l_ds_copy.sort_values(by=arrange_by, inplace=True)

    if group_by is None:
        group_by = []

    for s in params['shift']:
        l_ds_copy[new_col_namespace % str(s)] = l_ds_copy.groupby(group_by)[value_col].shift(periods=s)
    
    if not inplace:
        return l_ds_copy

def add_lagged_means(l_ds, params = { 'group_by': None, 'arrange_by': None, 'value_col': None, 'shift': [1], 'col_namespace': 'lag_%sh'}, inplace = True ):
    group_by = params['group_by']
    arrange_by = params['arrange_by']
    value_col = params['value_col']
    new_col_namespace = params['col_namespace']

    assert value_col is not None, 'The value_col parameter need to be specified!'
    assert new_col_namespace is not None and (new_col_namespace % str(1)) != new_col_namespace, 'The col_namespace parameter need to be specified and contain \%\s special character!'
    
    if inplace == True:
        l_ds_copy = l_ds
    else:
        l_ds_copy = l_ds.copy()

    if arrange_by is not None:
        l_ds_copy.sort_values(by=arrange_by, inplace=True)

    if group_by is None:
        group_by = []

    for s in params['shift']:
        l_ds_copy[new_col_namespace % str(s)] = l_ds_copy.groupby(group_by)[value_col].transform(lambda x: x.shift(s).rolling(window=s, min_periods=1).mean())

    
    if not inplace:
        return l_ds_copy

def add_lagged_sums(l_ds, params = { 'group_by': None, 'arrange_by': None, 'value_col': None, 'shift': [1], 'col_namespace': 'lag_%sh'}, inplace = True ):
    group_by = params['group_by']
    arrange_by = params['arrange_by']
    value_col = params['value_col']
    new_col_namespace = params['col_namespace']

    assert value_col is not None, 'The value_col parameter need to be specified!'
    assert new_col_namespace is not None and (new_col_namespace % str(1)) != new_col_namespace, 'The col_namespace parameter need to be specified and contain \%\s special character!'
    
    if inplace == True:
        l_ds_copy = l_ds
    else:
        l_ds_copy = l_ds.copy()

    if arrange_by is not None:
        l_ds_copy.sort_values(by=arrange_by, inplace=True)

    if group_by is None:
        group_by = []

    for s in params['shift']:
        l_ds_copy[new_col_namespace % str(s)] = l_ds_copy.groupby(group_by)[value_col].apply(lambda x: x.rolling(s, min_periods = 1).sum()).reset_index(drop=True)
    
    if not inplace:
        return l_ds_copy