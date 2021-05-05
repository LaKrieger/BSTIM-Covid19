import os
import pandas as pd
import re

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm

from datetime import datetime, timedelta, date

import theano
import theano.tensor as tt
import pickle as pkl

from scipy import stats

import scipy as sp
print(f"Running on PyMC3 v{pm.__version__}")



def preprocess_LKOS_data(filename="Fallzahlen pro Tag.xlsx", outputname="preprocessedLKOS.csv", ID_to_name=False):
    cd = os.getcwd()  # has to be adapted for final data structure
    file = cd + "/" + filename
    data = pd.read_excel(file)  # , encoding='latin-1')
    data

    # change names to ids and drop natural names (but save them for later)
    data["ID"] = list(range(len(data.index)))
    data.set_index("ID", drop=True, inplace=True)
    id_to_name = data[["Bestätigte (Neu-)Fälle pro Tag"]]
    id_to_name.rename(columns={"Bestätigte (Neu-)Fälle pro Tag": "NL Name"}, inplace=True)
    data.drop(["Bestätigte (Neu-)Fälle pro Tag", "Summe"], axis=1, inplace=True)

    data.columns = pd.to_datetime(data.columns, dayfirst=True)
    df = data.transpose()

    df.to_csv(cd + "/" + outputname)
    if ID_to_name: id_to_name.to_csv(cd + "/ID_to_name.csv")
    print("Successfully saved newest data in appropriate form.")

def split_data(
        data,
        train_start,
        test_start,
        post_test
):
    """
        split_data(data,data_start,train_start,test_start)

    Utility function that splits the dataset into training and testing data as well as the corresponding target values.

    Returns:
    ========
        data_train:     training data (from beginning of records to end of training phase)
        target_train:   target values for training data
        data_test:      testing data (from beginning of records to end of testing phase = end of records)
        target_test:    target values for testing data
    """
    # print("\ntrain_start", train_start, "\ntest_start", test_start, "\npost test",post_test)
    target_train = data.loc[(train_start <= data.index)
                            & (data.index < test_start)]
    target_test = data.loc[(test_start <= data.index)
                           & (data.index < post_test)]

    data_train = data.loc[data.index < test_start]
    data_test = data

    return data_train, target_train, data_test, target_test


def load_data_n_weeks(
        csv_path,
        seperator=",",
        pad=None
):
    ''' loads the data starting at a given timepoint
    Arguments:
        start (int): Days after '2020-03-05' to start the data (adapted for new date in LKOS data) NOT ENTIRELY SURE WHY WE WOULD START LATER TBH
        csv_path (str): Path to the file, inclusing the file name
        pad (int): How many days are going to be added (nan filled) at the end
    Returns:
        data (pd.df): Daframe with date as index, columns with countie IDs and values in cells.
    '''

    data = pd.read_csv(csv_path, sep=seperator, encoding='iso-8859-1', index_col=0)

    data.index = [pd.Timestamp(date) for date in data.index]

    start = data.index[-1] - pd.Timedelta(days=56)

    data = data.loc[start <= data.index]  # changed start_day to start

    if pad is not None:
        last_date = data.index[-1]

        extended_index = pd.date_range(last_date + pd.Timedelta(days=1),
                                       last_date + pd.Timedelta(days=pad))
        for x in extended_index:
            data = data.append(pd.Series(name=x))

    data.index = [pd.Timestamp(date) for date in data.index]

    return data

def sample_x_days_incidence_by_county(samples, x):
    num_sample = len(samples)
    timesteps = len(samples[0])
    counties = len(samples[0][0])
    incidence = np.empty((num_sample, timesteps - x, counties), dtype="int64")
    for sample in range(num_sample):
        for week in range(timesteps - x):
            incidence[sample][week] = np.sum(samples[sample][week : week + x], axis=0)
    return incidence

def spatio_temporal_feature(times, locations):
    _times = [datetime.strptime(d, "%Y-%m-%d") for d in times]
    return np.asarray(_times).reshape((-1, 1)), np.asarray(locations).reshape((1, -1)).astype(np.float32)


def temporal_polynomial_feature(t0, t, tmax, order):
    #print("Aus report temporal polynomial feauture", t, t0, tmax, order)
    t = datetime.strptime(t, "%Y-%m-%d")
    t0 = datetime.strptime(t0, "%Y-%m-%d")
    tmax = datetime.strptime(tmax, "%Y-%m-%d")
    scale = (tmax - t0).days

    t_delta = (t - t0).days / scale
    #print(scale)
    #print(t_delta)
    #print(t_delta ** order)
    return t_delta ** order


# TemporalFourierFeature(SpatioTemporalFeature)

def temporal_periodic_polynomial_feature(t0, t, period, order):
    t = datetime.strptime(t, "%Y-%m-%d")
    t0 = datetime.strptime(t0, "%Y-%m-%d")
    tdelta = (t - t0).days % period

    return (tdelta / period) ** order


def temporal_sigmoid_feature(t0, t, scale):
    # what does scale do here?
    t = datetime.strptime(t, "%Y-%m-%d")
    t0 = datetime.strptime(t0, "%Y-%m-%d")
    t_delta = (t - t0) / scale
    return sp.special.expit(t_delta.days + (t_delta.seconds / (3600 * 24)))


# def report_delay_polynomial_feature(t0, t, t_max, order):
#     #print("Aus report delay polynomial feauture",t, t0, t_max, order)
#     t = datetime.strptime(t, "%Y-%m-%d")
#     t0 = datetime.strptime(t0, "%Y-%m-%d")
#     t_max = datetime.strptime(t_max, "%Y-%m-%d")
#     scale = (t_max - t0).days
#     _t = 0 if t <= t0 else (t - t0).days / scale
#     return _t ** order


def features(trange, order, demographic, periodic_poly_order=2, trend_poly_order=2, include_temporal=True, #trend_poly_order was 2
             include_periodic=True,
             include_demographics=True, include_report_delay=False):
    report_delay_order = order
    feature_collection = {
        "temporal_trend": {
            "temporal_polynomial_{}".format(i): temporal_polynomial_feature(
                trange[0], trange[1], trange[2], i
            )
            for i in range(trend_poly_order + 1)
        }
        if include_temporal
        else {},
        "temporal_seasonal": {
            "temporal_periodic_polynomial_{}".format(
                i
            ): temporal_periodic_polynomial_feature(trange[0], trange[1], 7, i)  # why 7
            for i in range(periodic_poly_order + 1)
        }
        if include_periodic
        else {},

        # "temporal_report_delay": {
        #     "report_delay": report_delay_polynomial_feature(
        #         trange[0], trange[1], trange[2], report_delay_order  #
        #     )
        # }
        # if include_report_delay
        # else {},
        "exposure": {
            "exposure": demographic * 1.0 / 100000
        }
    }

    return feature_collection


def datetimeadaptions(date):  # I don't like myself for doing this
    year = str(date)[:4]
    month = str(date)[5:7]
    day = str(date)[8:10]
    return year + "-" + month + "-" + day


def evaluate_features(days, counties, demographic, polynom_order):
    all_features = pd.DataFrame()

    first_day_train = datetimeadaptions(days[0])
    last_day_train = datetimeadaptions(days[-1])

    for day in days:
        trange = [first_day_train, datetimeadaptions(day), last_day_train]  # last days train oder last_day_forecast?

        for i, county in enumerate(counties):
            feature = features(trange, polynom_order, demographic[i],
                               include_temporal=True, include_periodic=True, include_demographics=True)
                               #include_report_delay=True, )

            feature['date'] = datetimeadaptions(day)
            feature['ID'] = county
            feature_df = pd.DataFrame.from_dict(feature)
            all_features = all_features.append(feature_df)

    return all_features


def get_features(target, demographics, poly_order=3):
    days, counties = target.index, target.columns
    # extract features

    all_features = evaluate_features(days, counties, demographics, polynom_order=poly_order)

    all_features.astype(float, errors='ignore')

    Y_obs = target.stack().values.astype(np.float32)

    len_targets = len(days) * len(counties)

    T_S = all_features.filter(regex="temporal_periodic_polynomial_\d", axis=0).dropna(
        axis=1)  # .values.astype(np.float32) #features["temporal_seasonal"].values.astype(np.float32)
    T_S = T_S.sort_values(["date", "ID"])
    T_S = T_S['temporal_seasonal'].to_numpy()
    T_S = T_S.reshape(len_targets, -1)

    T_T = all_features.filter(regex="temporal_polynomial_\d", axis=0).dropna(
        axis=1)  # features["temporal_trend"].values.astype(np.float32)
    T_T = T_T.sort_values(["date", "ID"])
    T_T = T_T["temporal_trend"].to_numpy()
    T_T = T_T.reshape(len_targets, -1)

    # T_D = all_features.filter(regex="report_delay", axis=0).dropna(
    #     axis=1)  # features["temporal_report_delay"].values.astype(np.float32)
    # T_D = T_D.sort_values(["date", "ID"])
    # T_D = T_D["temporal_report_delay"].to_numpy()
    # T_D = T_D.reshape(len_targets, -1)

    exposure = all_features.filter(regex="exposure", axis=0).dropna(
        axis=1)  # features["spatiotemporal"].values.astype(np.float32)
    exposure = exposure.sort_values(["date", "ID"])
    exposure = exposure["exposure"].to_numpy()
    exposure = exposure.reshape(len_targets, -1)

    # has to be sorted I guess? order matches the one of Y_obs =)
    #return [Y_obs, T_S, T_T, T_D, exposure]
    return [Y_obs, T_S, T_T, exposure]

def make_model(features, data):
    target = features[0]

    T_S = features[1]
    T_T = features[2]
    # T_D = features[3]
    exposure = features[3] #was 4
    days, counties = data.index, data.columns

    log_exposure = np.log(exposure).astype(np.float64).ravel()
    num_obs = np.prod(target.shape)
    num_t_s = T_S.shape[1]
    num_t_t = T_T.shape[1]
    #num_t_d = T_D.shape[1]
    num_counties = len(counties)
    with pm.Model() as model:
        # priors
        # δ = 1/√α
        δ = pm.HalfCauchy("δ", 10, testval=1.0)
        α = pm.Deterministic("α", np.float64(1.0) / δ)

        W_t_s = pm.Normal(
            "W_t_s", mu=0, sd=10, testval=np.zeros(num_t_s), shape=num_t_s
        )
        W_t_t = pm.Normal(
            "W_t_t",
            mu=0,
            sd=10,
            testval=np.zeros((num_counties, num_t_t)),
            shape=(num_counties, num_t_t),
        )

        # W_t_d = pm.Normal(
        #     "W_t_d", mu=0, sd=10, testval=np.zeros(num_t_d), shape=num_t_d
        # )
        #         W_ts = pm.Normal(
        #             "W_ts", mu=0, sd=10, testval=np.zeros(num_ts), shape=num_ts
        #         )

        #param_names = ["δ", "W_t_s", "W_t_t", "W_t_d"]  # , "W_ts"]
        #params = [δ, W_t_s, W_t_t, W_t_d]  # , W_ts]

        expanded_Wtt = tt.tile(
            W_t_t.reshape(shape=(1, num_counties, -1)), reps=(28, 1, 1)

        )
        expanded_TT = np.reshape(T_T, newshape=(28, num_counties, -1))

        result_TT = tt.flatten(tt.sum(expanded_TT * expanded_Wtt, axis=-1))

        # calculate mean rates
        μ = pm.Deterministic(
            "μ",
            tt.exp(
                tt.dot(T_S, W_t_s)
                + result_TT
                #+ tt.dot(T_D, W_t_d)
                # + tt.dot(TS, W_ts)
                + log_exposure
            ),
        )
        # constrain to observations
        pm.NegativeBinomial("Y", mu=μ, alpha=α, observed=target)
    return model


def sample_parameters(
        target,
        n_init=100,
        samples=1000,
        chains=2,
        init="advi",
        target_accept=0.8,
        max_treedepth=10,
        cores=2,
        **kwargs
):
    """
        sample_parameters(target, samples=1000, cores=8, init="auto", **kwargs)

    Samples from the posterior parameter distribution, given a training dataset.
    The basis functions are designed to be causal, i.e. only data points strictly
    predating the predicted time points are used (this implies "one-step-ahead"-predictions).
    """

    # self.init_model(target)

    with model:
        # run!
        nuts = pm.step_methods.NUTS(
            # vars= params,
            target_accept=target_accept,
            max_treedepth=max_treedepth,
        )
        trace = pm.sample(
            samples,
            nuts,
            chains=chains,
            cores=cores,
            compute_convergence_checks=False,
            **kwargs
        )
    return trace


def sample_predictions(
        target_days_counties,
        demographics,
        parameters,
        prediction_days,
        average_periodic_feature=False,
        average_all=False,
        init="auto",
):
    PPO = 2
    TPO = 2 # was 2
    target_days = target_days_counties.index
    prediction_length = len(target_days) + len(prediction_days)


    target_counties = target_days_counties.columns
    num_counties = len(target_counties)

    all_days = target_days.append(prediction_days)

    all_days_counties = pd.DataFrame(index=all_days, columns=target_counties)

    # extract features
    features_ = get_features(all_days_counties, demographics)
    target = features_[0]
    T_S = features_[1]
    T_T = features_[2]
    #T_D = features_[3]
    exposure = features_[3]

    log_exposure = np.log(exposure).astype(np.float64).ravel()

    if average_periodic_feature:
        T_S = np.reshape(T_S, newshape=(-1, num_counties, 5))
        mean = np.mean(T_S, axis=0, keepdims=True)
        T_S = np.reshape(np.tile(mean, reps=(T_S.shape[0], 1, 1)), (-1, 5))

    if average_all:
        T_S = np.reshape(T_S, newshape=(prediction_length, num_counties, -1))
        mean = np.mean(T_S, axis=0, keepdims=True)
        T_S = np.reshape(np.tile(mean, reps=(prediction_length, 1, 1)), (-1, PPO + 1))  # periodic feature!!!


        # T_D = np.reshape(T_D, newshape=(47, num_counties, -1))
        # mean = np.mean(T_D, axis=0, keepdims=True)
        # T_D = np.reshape(np.tile(mean, reps=(47, 1)), newshape=(-1, 1))

        log_exposure = np.reshape(log_exposure, newshape=(prediction_length, num_counties))
        mean = np.mean(log_exposure, axis=0, keepdims=True)
        log_exposure = np.reshape(np.tile(mean, reps=(prediction_length, 1, 1)), (-1))


    # extract coefficient samples
    α = parameters["α"]
    W_t_s = parameters["W_t_s"]
    W_t_t = parameters["W_t_t"]
    #W_t_d = parameters["W_t_d"]
    # W_ts = parameters["W_ts"]
    #print("This is W-t-t", W_t_t)
    num_predictions = len(target_days) * len(target_counties) + len(
        prediction_days
    ) * len(target_counties)
    num_parameter_samples = α.size

    y = np.zeros((num_parameter_samples, num_predictions), dtype=np.float64)
    μ = np.zeros((num_parameter_samples, num_predictions), dtype=np.float64)

    expanded_Wtt = np.tile(
        np.reshape(W_t_t, newshape=(-1, 1, num_counties, TPO + 1)), reps=(1, prediction_length, 1, 1)
    )

    expanded_TT = np.reshape(T_T, newshape=(1, prediction_length, num_counties, TPO + 1))  # TT=1


    result_TT = np.reshape(
        np.sum(expanded_TT * expanded_Wtt, axis=-1), newshape=(-1,prediction_length * num_counties)
    )


    for i in range(num_parameter_samples):
        if i % 100 == 0: print(i, "/", num_parameter_samples)
        μ[i, :] = np.exp(
            np.dot(T_S, W_t_s[i])
            + result_TT[i]
            #+ np.dot(T_D, W_t_d[i])
            + log_exposure
        )

        y[i, :] = pm.NegativeBinomial.dist(mu=μ[i, :], alpha=α[i]).random()
    print("y", y, "μ", μ, "α", α)
    return {"y": y, "μ": μ, "α": α}


def get_change(current, previous):
    '''
    Get change in percent between two values. Sign indicates increase or decrease (-)
    '''
    result = 0
    if current == previous:
        return 0
    try:
        result = (abs(current - previous) / previous) * 100.0
    except ZeroDivisionError:
        return float('inf')
    if current < previous:
        return -result
    else:
        return result


def get_percent(this_many, of_all):
    # calculates percent and rounds to three positions after the comma for readibility
    result = this_many / of_all * 100
    return round(result, 3)


def all_changes(liste):
    # helper function to gett all percent changes
    changes = []
    for i in liste:
        changes.append(get_percent(i, np.sum(liste)))
    return changes


def count_range_in_list(li, min, max):
    # counts how many occurences of values in a range there are in a list
    ctr = 0
    for x in li:
        if min <= x <= max:
            ctr += 1
    return ctr


def all_ranges(liste):
    amount = len(liste)  # should be 2000, just to make sure
    df = pd.DataFrame(index=["absolut", "percent"])

    # collects all ranges
    # this could have been done in a nice loop in 2 lines but I don't want to think
    df["Fallen insgesamt"] = [count_range_in_list(liste, -float('inf'), 0),
                              get_percent(count_range_in_list(liste, -float('inf'), 0), amount)]
    df["Fallen > 10 %"] = [count_range_in_list(liste, -float('inf'), -10),
                           get_percent(count_range_in_list(liste, -float('inf'), -10), amount)]
    df["Fallen > 20 %"] = [count_range_in_list(liste, -float('inf'), -20),
                           get_percent(count_range_in_list(liste, -float('inf'), -20), amount)]
    df["Fallen > 30 %"] = [count_range_in_list(liste, -float('inf'), -30),
                           get_percent(count_range_in_list(liste, -float('inf'), -30), amount)]
    df["Fallen > 40 %"] = [count_range_in_list(liste, -float('inf'), -40),
                           get_percent(count_range_in_list(liste, -float('inf'), -40), amount)]
    df["Fallen > 50 %"] = [count_range_in_list(liste, -float('inf'), -50),
                           get_percent(count_range_in_list(liste, -float('inf'), -50), amount)]
    df["Fallen > 60 %"] = [count_range_in_list(liste, -float('inf'), -60),
                           get_percent(count_range_in_list(liste, -float('inf'), -60), amount)]
    df["Fallen > 70 %"] = [count_range_in_list(liste, -float('inf'), -70),
                           get_percent(count_range_in_list(liste, -float('inf'), -70), amount)]
    df["Fallen > 80 %"] = [count_range_in_list(liste, -float('inf'), -80),
                           get_percent(count_range_in_list(liste, -float('inf'), -80), amount)]
    df["Fallen > 90 %"] = [count_range_in_list(liste, -float('inf'), -90),
                           get_percent(count_range_in_list(liste, -float('inf'), -90), amount)]  # arbitrary large value
    df["Steigen insgesamt"] = [count_range_in_list(liste, 0, float('inf')),
                               get_percent(count_range_in_list(liste, 0, float('inf')), amount)]
    df["Steigen > 10 %"] = [count_range_in_list(liste, 10, float('inf')),
                            get_percent(count_range_in_list(liste, 10, float('inf')), amount)]
    df["Steigen > 20 %"] = [count_range_in_list(liste, 20, float('inf')),
                            get_percent(count_range_in_list(liste, 20, float('inf')), amount)]
    df["Steigen > 30 %"] = [count_range_in_list(liste, 30, float('inf')),
                            get_percent(count_range_in_list(liste, 30, float('inf')), amount)]
    df["Steigen > 40 %"] = [count_range_in_list(liste, 40, float('inf')),
                            get_percent(count_range_in_list(liste, 40, float('inf')), amount)]
    df["Steigen > 50 %"] = [count_range_in_list(liste, 50, float('inf')),
                            get_percent(count_range_in_list(liste, 50, float('inf')), amount)]
    df["Steigen > 60 %"] = [count_range_in_list(liste, 60, float('inf')),
                            get_percent(count_range_in_list(liste, 60, float('inf')), amount)]
    df["Steigen > 70 %"] = [count_range_in_list(liste, 70, float('inf')),
                            get_percent(count_range_in_list(liste, 70, float('inf')), amount)]
    df["Steigen > 80 %"] = [count_range_in_list(liste, 80, float('inf')),
                            get_percent(count_range_in_list(liste, 80, float('inf')), amount)]
    df["Steigen > 90 %"] = [count_range_in_list(liste, 90, float('inf')),
                            get_percent(count_range_in_list(liste, 90, float('inf')), amount)]  # arbitrary large value

    df = df.transpose()
    fallen = df[["percent"]].iloc[0:10]
    fallen.columns = ["Sinkende Infektionszahlen"]
    steigen = df[["percent"]].iloc[10:]
    steigen.columns = ["Ansteigende Infektionszahlen"]

    ticks = ["Gesamt", " mehr als 10 %", " mehr als 20 %", " mehr als 30 %", ">40%", ">50%", ">60%", ">70%", ">80%",
             ">90%"]
    fallen["name"] = ticks
    steigen["name"] = ticks
    fallen.set_index("name", inplace=True)

    steigen.set_index("name", inplace=True)

    return steigen, fallen






if __name__ == "__main__":
    preprocess_LKOS_data(filename="Fallzahlen 05.05.21.xlsx")
    days_into_future = 5 #14 # 5
    number_of_weeks = 4

    add_info_pd = pd.read_csv("ID_to_name_demographic.csv")
    additional_info = add_info_pd.to_dict("records")
    demographic = add_info_pd["demographic"].to_numpy()
    nl_names = add_info_pd["NL Name"].to_numpy()

    data = load_data_n_weeks("preprocessedLKOS.csv", pad=days_into_future)
    start_day = data.index[-days_into_future] - pd.Timedelta(days=number_of_weeks * 7)
    data[data < 0] = 0 # to get rid of the negative values
    print(start_day)
    today = date.today()# - pd.Timedelta(days=2)
    print(today)
    today_print = today.strftime("%b-%d-%Y")
    print(today_print)

    data_train, target_train, data_test, target_test = split_data(
        data,
        train_start=start_day,
        test_start=start_day + pd.Timedelta(days=number_of_weeks * 7),
        post_test=start_day + pd.Timedelta(days=number_of_weeks * 7 + days_into_future),
    )

    features_for_model = get_features(target_train, demographic)
    model = make_model(features_for_model, target_train)
    trace = sample_parameters(target_train, chains=2, cores=12)  # , samples = 50) #for superficial testing
    with open("{}_trace_{}".format(str(today), days_into_future), "wb") as f:
        pkl.dump(trace, f)
    with open("{}_model_{}".format(str(today), days_into_future), "wb") as f:
        pkl.dump(model, f)

    pred = sample_predictions(
        target_train,
        demographic,
        trace,
        target_test.index,
        average_periodic_feature=False,
        average_all=False,
    )
    with open("{}_predictions_{}".format(str(today), days_into_future), "wb") as f:
        pkl.dump(pred, f)

    pred_trend = sample_predictions(
        target_train,
        demographic,
        trace,
        target_test.index,
        average_periodic_feature=False,
        average_all=True,
    )
    with open("{}_predictions_trend_{}".format(str(today), days_into_future), "wb") as f:
        pkl.dump(pred_trend, f)

    ## from here on we plot!

    data = load_data_n_weeks("preprocessedLKOS.csv")

    with open("{}_model_{}".format(str(today), days_into_future), "rb") as f:
        model = pkl.load(f)
    with open("{}_trace_{}".format(str(today), days_into_future), "rb") as f:
        trace = pkl.load(f)
    with open("{}_predictions_{}".format(str(today), days_into_future), "rb") as f:
        pred = pkl.load(f)
    with open("{}_predictions_trend_{}".format(str(today), days_into_future), "rb") as f:
        pred_trend = pkl.load(f)

    start_day = data.index[-1] - pd.Timedelta(days=28)
    test = data.index[-1]


    day_0 = data.index[-1]
    day_m5 = day_0 - pd.Timedelta(days=5)
    day_p5 = day_0 + pd.Timedelta(days = days_into_future)

    # target_counties = data_selection.columns
    target_counties = data.columns
    num_counties = len(target_counties)

    _, target, _, _ = split_data(
        data, train_start=start_day, test_start=day_0, post_test=day_p5
    )
    # print(target)
    ext_index = pd.date_range(start_day, day_p5)

    # changed res to pred and res_trend to pred_trend

    prediction_samples_trend = np.reshape(
        pred_trend["y"], (pred_trend["y"].shape[0], -1, num_counties)
    )
    prediction_quantiles_trend = np.quantile(prediction_samples_trend, [0.05, 0.25, 0.75, 0.95], axis=0)
    prediction_q25_trend = pd.DataFrame(
        data=prediction_quantiles_trend[1], index=ext_index, columns=target.columns
    )
    prediction_q75_trend = pd.DataFrame(
        data=prediction_quantiles_trend[2], index=ext_index, columns=target.columns
    )
    prediction_q5_trend = pd.DataFrame(
        data=prediction_quantiles_trend[0], index=ext_index, columns=target.columns
    )
    prediction_q95_trend = pd.DataFrame(
        data=prediction_quantiles_trend[3], index=ext_index, columns=target.columns
    )

    prediction_samples_trend_mu = np.reshape(
        pred_trend["μ"], (pred_trend["μ"].shape[0], -1, num_counties)
    )
    prediction_mean_trend = pd.DataFrame(
        data=np.mean(prediction_samples_trend_mu, axis=0),
        index=ext_index,
        columns=target.columns,
    )

    IDNameDem = additional_info

    # colors for curves
    C1 = "#D55E00"
    C2 = "#E69F00"
    C3 = "#0073CF"

    # quantiles we want to plot
    qs = [0.25, 0.50, 0.75, 0.95]

    i_start_day = (start_day - data.index.min()).days

    county_ids = target.columns

    # Load our prediction samples
    res = pred
    n_days = (day_p5 - start_day).days

    prediction_samples = prediction_samples[:, i_start_day:i_start_day + n_days, :]

    first_week = prediction_samples_trend_mu[:, :7, :]
    second_week = prediction_samples_trend_mu[:, 7:14, :]
    third_week = prediction_samples_trend_mu[:, 14:21, :]
    forth_week = prediction_samples_trend_mu[:, 21:28, :]
    next_five_days = prediction_samples_trend_mu[:, 28:, :]

    cd = os.getcwd()

    plots = cd + "\\Plots_{}weeksPast_".format(number_of_weeks) + "{}daysFuture\\".format(days_into_future)
    currentplots = plots + str(today)
    try:
        os.mkdir(plots)
    except OSError:
        print("Creation of the directory %s failed" % plots)
    else:
        print("Successfully created the directory %s " % plots)

    try:
        os.mkdir(currentplots)
    except OSError:
        print("Creation of the directory %s failed" % currentplots)
    else:
        print("Successfully created the directory %s " % currentplots)

    dates = [pd.Timestamp(day) for day in ext_index]
    days = [(day - min(dates)).days for day in dates]


    for county_id in range(len(data.columns)):

        county_id = str(county_id)

        fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(15, 15))
        fig.suptitle(IDNameDem[int(county_id)].get("NL Name"))

        ax1 = plt.subplot(2, 2, (1, 2))
        ax1.plot_date(dates[:-days_into_future],
                      target[county_id],
                      color="k",
                      label="Bestätigte Infektionen - {}".format(IDNameDem[int(county_id)].get("NL Name")))
        ax1.plot_date(
            dates,
            prediction_mean_trend[county_id],
            "-",
            color=C1,
            label="Durchschnittsvorhersage der 2000 Modelle",
            linewidth=2.0,
            zorder=4)
        ax1.fill_between(
            dates,
            prediction_q25_trend[county_id],
            prediction_q75_trend[county_id],
            facecolor=C2,
            alpha=0.5,
            zorder=1)
        ax1.plot_date(
            dates,
            prediction_q25_trend[county_id],
            ":",
            color=C2,
            label="Q25 - Die meisten Vorhersagen liegen in diesem Bereich",
            linewidth=2.0,
            zorder=3)
        ax1.plot_date(dates,  # upper line
                      prediction_q95_trend[county_id], ":",
                      label="Q05-Q95 - Einige Vorhersagen liegen in diesem Bereich",
                      color="green", alpha=0.5, linewidth=2.0, zorder=1)
        ax1.axvline(data.index[-1], label='Tag der Datenerhebung')
        ax1.legend(loc="upper left")  # everything above will be included in legend
        ax1.fill_between(
            dates,
            prediction_q5_trend[county_id],
            prediction_q95_trend[county_id],
            facecolor="green",
            alpha=0.25,
            zorder=0)
        ax1.plot_date(dates,  # lower line
                      prediction_q5_trend[county_id],
                      ":",
                      color="green", alpha=0.5, linewidth=2.0, zorder=1)
        ax1.plot_date(  # upper of q25
            dates,
            prediction_q75_trend[county_id],
            ":",
            color=C2,
            linewidth=2.0,
            zorder=3)

        ax1.xaxis.set_tick_params(rotation=30, labelsize=10)
        ax1.set_ylabel("Anzahl Infektionen")
        highest_value = target[county_id].max()
        ax1.set_ylim(0, (int(highest_value) * 3))

        i = int(county_id)
        Precise_3to4week = np.zeros(
            shape=(37, 2000))  # to collect the precise change of all 2000 predictions from week 3 to 4
        mittelwert3Woche = prediction_mean_trend.to_numpy()[14:21,
                           i].mean()  # mean vorhersage für die 3. Woche für diese Kommune
        for k, j in enumerate(forth_week[:, :, i].mean(axis=1)):  # take mean of week 4 of the 2000 predictions
            Precise_3to4week[i, k] = get_change(j, mittelwert3Woche)
        steig, fall = all_ranges(Precise_3to4week[i])

        xPoints = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        xPoints2 = np.array([13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
        PointsforLine = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        PointsforLine2 = np.array([12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
        increasing = steig["Ansteigende Infektionszahlen"].tolist()  # [1:]#.reverse()
        decreasing = fall["Sinkende Infektionszahlen"].tolist()  # [1:]
        decreasing = decreasing[::-1]
        bottom = np.array([0 for i in range(len(PointsforLine))])

        ax2 = plt.subplot(2, 2, (3, 4))

        ax2.bar(xPoints, decreasing, color="forestgreen", label="Modelle, die ein Sinken \n > x % prognostizieren")
        horiz_line_data_sink = np.array([fall["Sinkende Infektionszahlen"][0] for i in range(len(PointsforLine))])
        ax2.plot(PointsforLine, horiz_line_data_sink, 'g--',
                 label="Modelle, die ein Sinken \n prognostizieren (insgesamt)")
        ax2.fill_between(PointsforLine, bottom, horiz_line_data_sink, alpha=0.5, color="forestgreen")

        ax2.bar(xPoints2, increasing, color="indianred", label="Modelle, die einen Anstieg \n > x % prognostizieren")
        horiz_line_data_steig2 = np.array(
            [steig["Ansteigende Infektionszahlen"][0] for i in range(len(PointsforLine2))])
        ax2.plot(PointsforLine2, horiz_line_data_steig2, 'r--',
                 label="Modelle, die einen Anstieg \n prognostizieren (insgesamt)")
        ax2.fill_between(PointsforLine2, bottom, horiz_line_data_steig2, alpha=0.5, color="indianred")

        ax2.set_xticks(np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]))
        ax2.set_xticklabels(["- > 90 %", "- > 80 %", "- > 70 %", "- > 60 %", "- > 50 %", "- > 40 %", "- > 30 %",
                             "- > 20 %", "- > 10 %", "- > 0 %", "+ > 0 %", '+ >10 %', '+ >20 %', '+ >30 %', '+ >40 %',
                             '+ >50 %', '+ >60 %',
                             '+ >70 %', '+ >80 %', '+ >90 %'], rotation=20)
        ax2.set_xlabel('Veraenderung Infektionszahlen')
        ax2.set_ylabel('Wahrscheinlichkeit für sinkende Zahlen')
        ax2.set_ylim(0, 100)
        ax2.legend(loc=1)
        ax3 = ax2.twinx()
        ax3.set_ylabel("Wahrscheinlichkeit für steigende Zahlen")

        plt.savefig(currentplots + "/{}_{}.png".format(today, IDNameDem[int(county_id)].get("NL Name")))
        plt.show()

