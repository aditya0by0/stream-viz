from flask import Flask, render_template
from temp import get_metric_score_list

app = Flask(__name__)


# Only for Testing - Please ignore
@app.route("/test")
def test():
    return render_template("Test api")


concept_drift_timepoints = [
    690,
    1381,
    2008,
    2635,
    3326,
    3953,
    4580,
    5303,
    5930,
    6557,
    7120,
    7811,
    8438,
    9129,
    9756,
    10383,
    11010,
    11701,
    12264,
    12827,
]
metric_score_list = get_metric_score_list()
# print(metric_score_list)


@app.route("/learn_drifts")
def learn_drifts():
    return concept_drift_timepoints, metric_score_list


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
