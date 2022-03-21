import pandas as pd
from tqdm import tqdm
import os
import time
import random
import sys
import google.api_core as google_core
from google.cloud import dialogflowcx_v3 as dialogflow_beta
from google.cloud.dialogflowcx_v3.services.sessions import SessionsClient
import uuid

PROJECT_ID = 'np-contactctrapps-dev'
OUTPUT = []
GOOGLE_END_POINT = "us-central1-dialogflow.googleapis.com" # "dialogflow.googleapis.com"


def detect_content_text(session,session_path,text):
    def gen_requests(session_path,text):
        """Generates requests for streaming.
        """
        try:
            text_input = dialogflow_beta.TextInput(text=text)
            query_input = dialogflow_beta.QueryInput(text=text_input, language_code="en-us")
            yield dialogflow_beta.StreamingDetectIntentRequest(query_input = query_input,
                session=session_path)
            # print("Yield audios to streaming analyze content.")
        except Exception as ex:
            print(ex)
    # print(text)
    # print(session_path)
    
    responses = session.streaming_detect_intent(requests=gen_requests(session_path,text),timeout=60)
    for response in responses:
        detected_text = ""
        for single in response.detect_intent_response.query_result.response_messages:
            if 'text' in single:
                detected_text = single.text
        return str(detected_text),str(response.detect_intent_response.query_result.intent.display_name),str(response.detect_intent_response.query_result.intent_detection_confidence)

# def detect_intent_texts(project_id, session_id, text, language_code):
#     """Returns the result of detect intent with texts as inputs.
#     Using the same `session_id` between requests allows continuation
#     of the conversation."""
#     try:
#         session = SESSION_CLIENT.session_path(project_id, session_id)
#         text_input = dialogflow.types.TextInput(text='!!!', language_code=language_code)
#         query_input = dialogflow.types.QueryInput(text=text_input)
#         response = SESSION_CLIENT.detect_intent(session=session, query_input=query_input)
#         text_input = dialogflow.types.TextInput(text=text, language_code=language_code)
#         query_input = dialogflow.types.QueryInput(text=text_input)
#         response = SESSION_CLIENT.detect_intent(session=session, query_input=query_input)
#         # print(response)
#         # print(qp)
#         return response.query_result.query_text, response.query_result.intent.display_name, response.query_result.intent_detection_confidence

#    except ConnectionError:
#        print('Connection Error In Client.')


if __name__ == '__main__':
    #print(os.listdir('.'))
    input_data = pd.read_csv('store_flow_data.csv')
    input_data = input_data.drop_duplicates()
    texts = input_data['utterance'].tolist()
    client_config_options = google_core.client_options.ClientOptions()
    client_config_options.api_endpoint  = GOOGLE_END_POINT
    session_client = SessionsClient(client_options=client_config_options)
    for i, text in tqdm(enumerate(texts)):
        """
        i: is a a unique idea for each utterance in order to establish a new session per API call, 
        ensuring we clear context
        text: utterance
        """
        try:
            time.sleep(random.randint(0,2))
            #session_client = SessionsClient(client_options=client_config_options)
            uuidstring = uuid.uuid4()
            session_path = "projects/np-contactctrapps-dev/locations/us-central1/agents/95d88e89-b691-407e-a92e-a7b047db226e/sessions/" + str(uuidstring)
           
            query, intent, score  = detect_content_text(session_client,session_path,"!!!")
           
            query, intent, score  = detect_content_text(session_client,session_path,text)
            # print(i, query, intent, score)
            OUTPUT.append((text, intent, score))
        except Exception as ex:
            # print(ex)
            OUTPUT.append((text,'error','error'))
    # print(OUTPUT)

    pandaOUTPUT = pd.DataFrame(
        OUTPUT, columns=['utterance', 'cx_intent', 'cx_score'])

    #pandaOUTPUT['df_intent'] = pandaOUTPUT['df_intent'].apply(
      #  lambda s: '_'.join(s.split()))  # fix the string so it lines up with our test label
    OUTPUT = pd.merge(input_data, pandaOUTPUT, on='utterance', how='left')
    # print(OUTPUT)
    OUTPUT.to_csv('CX_output.csv', index=False)
    
