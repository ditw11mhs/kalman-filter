import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pandas import DataFrame as df


class FrontEndUtils:
    def filter_plot(self, process_output):
        plot_df = df(process_output)

        output_fig = px.line(
            plot_df,
            x="Time",
            y=[
                "Error Estimation",
                "Angle Estimation",
                "Zero Order Sensor",
                "Gyroscope Angle",
            ],
        )
        gain_fig = px.line(plot_df, x="Iteration", y=["Kalman Gain 1", "Kalman Gain 2"])
        st.markdown("### Comparing Output")
        st.plotly_chart(output_fig, use_container_width=True)
        st.markdown("### Gain Update")
        st.plotly_chart(gain_fig, use_container_width=True)

    def plot_best_kalman(self, best_kalman):
        z = best_kalman["RMSE"]
        best_kalman_df = df({"n": best_kalman["n"], "e": best_kalman["e"]})
        fig = go.Figure(
            data=[go.Surface(z=z, x=best_kalman_df["n"], y=best_kalman_df["e"])]
        )
        fig.update_layout(xaxis_title="n", yaxis_title="e")
        st.plotly_chart(fig, use_container_width=True)


class BackEndUtils:
    def padding(self, x):
        log = np.log2(len(x))
        return np.pad(
            x, (0, int(2 ** ((log - log % 1) + 1) - len(x))), mode="constant"
        ).flatten()

    def FFT(self, x):

        if np.log2(len(x)) % 1 > 0:
            x = self.padding(x)

        x = np.asarray(x, dtype=float)
        N = x.shape[0]

        N_min = min(N, 2)

        # DFT on all length-N_min sub-problems at once
        n = np.arange(N_min)
        k = n[:, None]
        W = np.exp(-2j * np.pi * n * k / N_min)
        X = np.dot(W, x.reshape((N_min, -1)))

        # Recursive calculation all at once
        while X.shape[0] < N:
            X_even = X[:, : int(X.shape[1] / 2)]
            X_odd = X[:, int(X.shape[1] / 2) :]
            factor = np.exp(-1j * np.pi * np.arange(X.shape[0]) / X.shape[0])[:, None]
            factor.shape, factor
            X = np.vstack([X_even + factor * X_odd, X_even - factor * X_odd])
        return X.flatten()

    @st.cache
    def load_data(self):
        raw_data = np.loadtxt("trial1.dat")
        data = {}
        data["Time"] = raw_data[:, 0].flatten()
        data["Zero Order Sensor"] = raw_data[:, 1].flatten()
        data["Gyroscope"] = raw_data[:, 3].flatten()
        data["Tilt Sensor"] = raw_data[:, 4].flatten()
        return data

    def rmse(self, y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))


class KalmanFilter(BackEndUtils):
    def __init__(self):
        self.data = self.load_data()
        self.dtm = self.data["Gyroscope"] - self.data["Tilt Sensor"]

    def fast_process(self, kalman_filter_state):

        A = np.array([[1, 1], [0, 1]])
        B = np.array([[1], [1]])

        n = kalman_filter_state["n"]
        e = kalman_filter_state["e"]

        post_error_covar = np.zeros((2, 2))
        kalman_gain = np.zeros((2, len(self.dtm)))
        d_theta_b_h_min = np.zeros((2, len(self.dtm) + 1))
        d_theta_b_h = np.zeros((2, len(self.dtm)))

        for i in np.arange(len(self.dtm)):

            # Priory Covar Calculation

            priori_error_covar = np.dot(A, np.dot(post_error_covar, A.T)) + B * B.T * e

            # Kalman Gain Calculation

            kalman_gain[:, i] = priori_error_covar[:, 0] / (
                priori_error_covar[0, 0] + n
            )

            # Kalman Filtering (Estimation)
            v = self.dtm[i] - d_theta_b_h_min[0, i]

            d_theta_b_h[:, i] = d_theta_b_h_min[:, i] + kalman_gain[:, i] * v

            # Post Covar Update
            post_copy = post_error_covar.copy()
            post_error_covar[0, :] = post_copy[0, :] * np.array([1, -kalman_gain[0, i]])
            post_error_covar[1, :] = np.dot(
                np.array([-kalman_gain[1, i], 1]), post_copy
            )

            # Prediction
            d_theta_b_h_min[:, i + 1] = np.dot(
                np.array([[1, 1], [0, 1]]), d_theta_b_h[:, i].reshape(-1, 1)
            ).flatten()
        return {
            "RMSE": self.rmse(
                self.data["Zero Order Sensor"],
                self.data["Gyroscope"] - d_theta_b_h[0, :],
            )
        }

    def process(self, kalman_filter_state):
        process_output = {}
        theta = self.data["Gyroscope"]

        # Input Difference
        dtm = self.data["Gyroscope"] - self.data["Tilt Sensor"]

        filter_output = self.kalman_filter(dtm, kalman_filter_state)
        process_output = filter_output
        process_output["Angle Estimation"] = theta - filter_output["Error Estimation"]
        process_output["Zero Order Sensor"] = self.data["Zero Order Sensor"]
        process_output["Gyroscope Angle"] = self.data["Gyroscope"]
        process_output["Tilt Sensor"] = self.data["Tilt Sensor"]
        process_output["RMSE"] = self.rmse(
            process_output["Zero Order Sensor"], process_output["Angle Estimation"]
        )
        return process_output

    def kalman_filter(self, input_filter, kalman_filter_state):
        # Initialize
        dtm = input_filter

        A = np.array([[1, 1], [0, 1]])
        B = np.array([[1], [1]])

        n = kalman_filter_state["n"]
        e = kalman_filter_state["e"]

        post_error_covar = np.zeros((2, 2))
        kalman_gain = np.zeros((2, len(dtm)))
        d_theta_b_h_min = np.zeros((2, len(dtm) + 1))
        d_theta_b_h = np.zeros((2, len(dtm)))

        for i in np.arange(len(dtm)):

            # Priory Covar Calculation

            priori_error_covar = np.dot(A, np.dot(post_error_covar, A.T)) + B * B.T * e

            # Kalman Gain Calculation

            kalman_gain[:, i] = priori_error_covar[:, 0] / (
                priori_error_covar[0, 0] + n
            )

            # Kalman Filtering (Estimation)
            v = dtm[i] - d_theta_b_h_min[0, i]

            d_theta_b_h[:, i] = d_theta_b_h_min[:, i] + kalman_gain[:, i] * v

            # Post Covar Update
            post_copy = post_error_covar.copy()
            post_error_covar[0, :] = post_copy[0, :] * np.array([1, -kalman_gain[0, i]])
            post_error_covar[1, :] = np.dot(
                np.array([-kalman_gain[1, i], 1]), post_copy
            )

            # Prediction
            d_theta_b_h_min[:, i + 1] = np.dot(
                np.array([[1, 1], [0, 1]]), d_theta_b_h[:, i].reshape(-1, 1)
            ).flatten()

        filter_output = {}
        filter_output["Kalman Gain 1"] = kalman_gain[0, :]
        filter_output["Kalman Gain 2"] = kalman_gain[1, :]
        filter_output["Error Estimation"] = d_theta_b_h[0, :]
        filter_output["Prediction"] = d_theta_b_h_min[0, :-1]
        filter_output["Time"] = self.data["Time"]
        filter_output["Iteration"] = np.arange(len(dtm))
        return filter_output

    def best_kalman_parameter_search(self, best_parameter_state):
        range_n_start = best_parameter_state["N Start"]
        range_n_end = best_parameter_state["N End"]
        range_e_start = best_parameter_state["E Start"]
        range_e_end = best_parameter_state["E End"]
        data_amount = best_parameter_state["Data Amount"]

        z_output = np.zeros((data_amount, data_amount))
        n = np.linspace(range_n_start, range_n_end, data_amount)
        e = np.linspace(range_e_start, range_e_end, data_amount)

        progress_bar = st.progress(0.0)
        amount = 0.0
        for i in range(data_amount):
            for j in range(data_amount):
                process_output = self.fast_process({"n": n[i], "e": e[j]})
                z_output[i, j] = process_output["RMSE"]
                amount += 1
                progress_bar.progress((amount) / (data_amount ** 2))
        # st.write(z_output)
        return {"n": n, "e": e, "RMSE": z_output.T}
