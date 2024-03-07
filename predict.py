from train import preprocess_image, prediction_with_cairo
from giza_actions.action import action, Action

@action(name=f'Prediction with Cairo', log_prints=True)
def prediction(MODEL_ID, VERSION_ID):
    image = preprocess_image("./imgs/three.png")
    (result, request_id) = prediction_with_cairo(image, MODEL_ID, VERSION_ID)
    print("Result: ", result)
    print("Request id: ", request_id)

    return result, request_id

if __name__=="__main__":
    prediction_action_deploy = Action(entrypoint=prediction, name="verifiable-pytorch-mnist-prediction")
    prediction_action_deploy.serve(name="verifiable-pytorch-mnist-prediction")

    # Update with correct values
    prediction(MODEL_ID=422, VERSION_ID=3)  