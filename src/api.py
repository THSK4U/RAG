from typing import Any

from flask import Flask, flash, redirect, render_template, request, url_for

from src.__main__ import full_pipeline

app = Flask(__name__)
app.secret_key = "RAG"


@app.route("/", methods=["POST", "GET"])
def post_query() -> Any:
    if request.method == "POST":
        try:
            query = request.form["query"]
            k_val = request.form.get("k")
            k = int(k_val) if k_val else 5

            result = full_pipeline(query=query, k=k)
            flash(result)

        except Exception as e:
            flash(f"{str(e)}", "error")
            return redirect(url_for("post_query"))

    return render_template("chat_ui.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7331, debug=True)
