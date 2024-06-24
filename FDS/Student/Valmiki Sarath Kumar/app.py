from flask import Flask, request, render_template
import random

app = Flask(__name__)

# Encode 'type' feature
types = {"CASH_IN": 0, "CASH_OUT": 1, "DEBIT": 2, "PAYMENT": 3, "TRANSFER": 4}

@app.route('/')
def form():
    return render_template('index.html')

@app.route('/process_form', methods=['POST'])
def process_form():
    # Get form data
    step = float(request.form['step'])
    type_value = request.form['type']
    amount = float(request.form['amount'])
    name_dest = request.form['nameDest']
    balance_diff_orig = float(request.form['balanceDiffOrig'])
    balance_diff_dest = float(request.form['balanceDiffDest'])

    # Encode 'type'
    type_encoded = types[type_value]

    # Prepare the input data
    input_data = [
        step,
        type_encoded,
        amount,
        name_dest,  # If nameDest is a string, it should be encoded or processed accordingly
        balance_diff_orig,
        balance_diff_dest
    ]

    # Simulate a model prediction (replace this with actual model loading and prediction code)
    # For now, assume a random prediction
    is_fraudulent = random.choice([True, False])

    # Display the result
    if is_fraudulent:
        result = "The transaction is predicted to be fraudulent."
    else:
        result = "The transaction is predicted to be non-fraudulent."

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
