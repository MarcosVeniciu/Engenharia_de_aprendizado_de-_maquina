from io import StringIO
from joblib import load
import pandas as pd
import Pyro5.api
import json



@Pyro5.api.expose
class ClassificationServer:

    def __init__(self):
        self.model = load('modelo_salvo.joblib')
        print("Modelo Funcionando!!")


    @Pyro5.api.expose
    def classificar(self, input_data):
        try:
            json_data = StringIO(input_data)
            df = pd.read_json(json_data)
            # Use the model to make a prediction on the input data
            prediction = self.model.predict(df)
            # Return the prediction result
            return json.dumps(prediction.tolist())
        except Exception as e:
            return f"Erro ao processar a entrada: {str(e)}"

if __name__ == '__main__':
    # Inicializa o servidor Pyro5
    daemon = Pyro5.api.Daemon()

    # Registra o objeto remoto no servidor Pyro5
    objeto_remoto = ClassificationServer()
    #uri = daemon.register(objeto_remoto)
    #uri = Pyro5.api.locate_ns(host="\"3.86.82.136\"", port=9090).register(objeto_remoto)
    uri = Pyro5.api.locate_ns(host="3.86.82.136", port=9090).register(objeto_remoto)

    # Obtém uma referência ao Name Server
    ns = Pyro5.api.locate_ns()

    # Registra a URI do objeto remoto no Name Server
    ns.register("servidor_AP7B", uri) # registra o name server como servidor

    # Inicia o servidor Pyro5
    print("Servidor aguardando conexões...")
    daemon.requestLoop()