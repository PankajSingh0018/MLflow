import mlflow

def calculate(x,y):
    z= x**y

    return z


if __name__=="__main__":

    # To start the ML flow server 
    with mlflow.start_run():
        # assigning the values to x,y 
        x,y = 10,3

        # calling the major function calculate 
        qz= calculate(x,y)

        # tracking the experiment 
        mlflow.log_param("x",x )
        mlflow.log_param("y",y)

        # tracking the  metric 
        mlflow.log_metric("qz",qz)

