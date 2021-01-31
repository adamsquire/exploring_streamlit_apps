import streamlit as st
import pickle
from datetime import datetime
startTime = datetime.now()

filename ="model.sv"
model = pickle.load(open(filename,'rb'))


sex_d = {0:"Female",1:"Male"}
pclass_d = {0:"First",1:"Second", 2:"Third"}
embarked_d = {0:"Cherbourg", 1:"Queenstown", 2:"Southampton"}

def main():

	st.set_page_config(page_title="Titanic Survival App")
	overview = st.beta_container()
	left, right = st.beta_columns(2)
	prediction = st.beta_container()

	with overview:
		st.title("Titanic App")
		st.markdown("Model: Random Forest Classifier, trained on titanic training dataset from [Kaggle](https://www.kaggle.com/c/titanic/data). See associated notebook for details on how the model was trained.")

	with left:
		sex_radio = st.radio( "Sex", list(sex_d.keys()), format_func=lambda x : sex_d[x] )
		pclass_radio = st.radio( "Ticket Class", list(pclass_d.keys()), format_func=lambda x: pclass_d[x])
		embarked_radio = st.radio( "Port of Embarkment", list(embarked_d.keys()), index=2, format_func= lambda x: embarked_d[x] )

	with right:
		age_slider = st.slider("Age", value=50, min_value=1, max_value=100)
		sibsp_slider = st.slider( "# Siblings / Spouses on board", min_value=0, max_value=8)
		parch_slider = st.slider( "# Parents / Children on board", min_value=0, max_value=6)
		fare_slider = st.slider( "Passenger Fare", min_value=0, max_value=500, step=10)


	data = [[pclass_radio, sex_radio,  age_slider, sibsp_slider, parch_slider, fare_slider, embarked_radio]]
	survival = model.predict(data)
	s_confidence = model.predict_proba(data)

	with prediction:
		st.header("Survived? {0}".format("Yes" if survival[0] == 1 else "No"))
		st.subheader("Confidence {0:.2f} %".format(s_confidence[0][survival][0] * 100))

if __name__ == "__main__":
    main()
