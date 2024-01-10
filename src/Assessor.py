import pandas as pd
import numpy as np

class Assessor:
    def __init__(self):
        pass

    def assess_sentiment(self, model, preprocessor, vectorizer, custom_input):
        # string -> dataframe
        df_custom_input = pd.DataFrame([custom_input], columns=['text'])
        # preprocessing
        df_custom_input['text'] = preprocessor.preprocess(df_custom_input['text'])
        # list -> string
        df_custom_input['text'] = df_custom_input['text'].apply(lambda x: ' '.join(x))
        # vectorizing
        X_custom = vectorizer.transform(df_custom_input['text'])
        # prediction
        y_custom = model.predict(X_custom).item()
        print('INPUT:', custom_input, 'PREDICTION:', y_custom)

        return y_custom


    def assess_sentiment_with_every_model(self, models, model_names, preprocessor, vectorizers, vectorizer_names, custom_input, label_encoders=None, label_encoder_names=None):
        # string -> dataframe
        df_custom_input = pd.DataFrame([custom_input], columns=['text'])
        # preprocessing
        df_custom_input['text'] = preprocessor.preprocess(df_custom_input['text'])
        # list -> string
        df_custom_input['text'] = df_custom_input['text'].apply(lambda x: ' '.join(x))
        

        # prediction for every model
        for model, model_name in zip(models, model_names):
            # vectorizing
            for v, vectorizer_name in zip(vectorizers, vectorizer_names):
                if vectorizer_name.startswith("basic") and model_name.endswith("basic"):
                    vectorizer = v
                if vectorizer_name.startswith("complex_classes") and model_name.endswith("complex_classes"):
                    vectorizer = v
                if vectorizer_name.startswith("conf_weights") and model_name.endswith("conf_weights"):
                    vectorizer = v
            X_custom = vectorizer.transform(df_custom_input['text'].copy())
            
            y_custom = model.predict(X_custom).item()
            
            # inverse transforming for xgb
            if model_name.startswith("xgb"):
                # label encoder
                for le, le_name in zip(label_encoders, label_encoder_names):
                    if le_name.startswith("basic") and model_name.endswith("basic"):
                        label_encoder = le
                    if le_name.startswith("complex_classes") and model_name.endswith("complex_classes"):
                        label_encoder = le
                    if le_name.startswith("conf_weights") and model_name.endswith("conf_weights"):
                        label_encoder = le
                y_custom = label_encoder.inverse_transform([y_custom])[0]

            print('MODEL:', model_name, 'INPUT:', custom_input, 'PREDICTION:', y_custom)
