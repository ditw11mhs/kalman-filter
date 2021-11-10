import streamlit as st
from backend_kalman_filter import FrontEndUtils, KalmanFilter


class Main(KalmanFilter, FrontEndUtils):
    def __init__(self):
        st.set_page_config(page_title="Kalman Filter", page_icon="📈")
        super().__init__()

    def main(self):

        st.title("Kalman Filter")
        st.markdown(
            """ 
                    By referring to course material in this week, design and realize a computer program for Kalman filtering to cut noise of gyroscope sensor by using fusion of gyro and tilt sensor.
                    
                    Use data of joint angle trajectories of normal gait (trial1.dat, column number/data: 1/time, 2/zero order sensor, 4/gyroscope, 5/tilt sensor).
                    
                    Do experiments to get best result of your result, and measure correlation filter output with zero order sensor in column 2.
                    
                    Submit your report of design and experiment and complete program."""
        )
        # st.write(self.data)
        st.markdown("## Kalman FIlter")
        with st.form("Kalman Filter"):
            kalman_filter_state = {}
            kalman_filter_state["n"] = st.number_input(
                "n", value=3.5, min_value=0.0, max_value=100.0
            )
            kalman_filter_state["e"] = st.number_input(
                "e", value=1.0, min_value=0.0, max_value=100.0
            )
            kalman_filter_state["Debug"] = st.checkbox("Debug")
            kalman_filter_state["Submit"] = st.form_submit_button("Submit")
            if kalman_filter_state["Submit"]:
                process_output = self.process(kalman_filter_state)
                # st.write(process_output)
                # self.filter_plot(filter_output)
        if kalman_filter_state["Submit"]:
            st.markdown(
                f"""
                        #### RMSE {process_output['RMSE']}"""
            )
            self.filter_plot(process_output)
            if kalman_filter_state["Debug"]:
                st.write(process_output)
        st.markdown("### Grid Search")
        with st.form("Best Param"):
            best_param_state = {}
            best_param_state["N Start"] = st.number_input(
                "N Start", min_value=0.0, max_value=1000.0
            )
            best_param_state["N End"] = st.number_input(
                "N End", min_value=0.0, max_value=1000.0
            )
            best_param_state["E Start"] = st.number_input(
                "E Start", min_value=0.0, max_value=1000.0
            )
            best_param_state["E End"] = st.number_input(
                "E End", min_value=0.0, max_value=1000.0
            )
            best_param_state["Data Amount"] = st.number_input(
                "Data Amount", value=100, min_value=0, max_value=1000
            )
            best_param_state["Submit"] = st.form_submit_button("Submit")
        if best_param_state["Submit"]:
            best_kalman = self.best_kalman_parameter_search(best_param_state)
            self.plot_best_kalman(best_kalman)


if __name__ == "__main__":
    main = Main()

    main.main()
