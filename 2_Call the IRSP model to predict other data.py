from IRSP_Model import IRSPModel
import pandas as pd
# Load the trained model
rf_new = IRSPModel()
rf_new.load_model("IRSP.pkl")

# Use the model to predict new data
new_data = pd.read_csv("YOUR CSV FILE.csv")
predictions = rf_new.predict(new_data)
pred_df = pd.DataFrame(predictions)
print("New data predict results:", predictions)