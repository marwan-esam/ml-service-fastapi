from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "postgresql://postgres:admin@localhost/test_db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# This is how the database sees a user
class DBUser(Base):
  __tablename__ = "users_v3"
  id = Column(Integer, primary_key=True)
  name = Column(String)
  age = Column(Integer)


class DBModel(Base):
  __tablename__ = "trained_models"

  id = Column(Integer, primary_key=True)
  status = Column(String)
  learning_rate = Column(Float)
  final_weight = Column(Float)
  final_bias = Column(Float)

# Tell the engine to build the tables
Base.metadata.create_all(bind=engine)

# This is how the API expects to receive data from the internet
class UserInput(BaseModel):
  name : str
  age : int

class TrainInput(BaseModel):
  learning_rate: float
  iterations: int
  x_data: list[float]
  y_data: list[float]


# API Routing

app = FastAPI()

@app.post("/add-user")
def add_new_user(incoming_data: UserInput):
  # Open a connection to the database
  session = SessionLocal()

  # Map the incoming API data to the database object
  new_user = DBUser(name=incoming_data.name, age=incoming_data.age)


  # Save it and close the connection
  session.add(new_user)
  session.commit()
  session.close()

  # Respond to the client
  return {
    "status": "Sucess",
    "message": f"User '{incoming_data.name}', age {incoming_data.age}, saved."
  }

@app.get("/users")
def get_all_users():
  session = SessionLocal()

  # Ask the database objects for every record inside the DBUser table
  all_users = session.query(DBUser).all()
  session.close()

  # Package the database objects into a list to return
  results = []
  for user in all_users:
    results.append({"id": user.id, "name": user.name, "age": user.age})
  return {
    "total_user": len(results),
    "users": results
  } 


def train_model_background(model_id: int, x_data: list[float], y_data: list[float], lr: float, iterations: int):
  weight = 0.0
  bias = 0.0
  n = len(x_data)

  for _ in range(iterations):
    weight_gradient = 0.0
    bias_gradient = 0.0

    for i in range(n):
      x = x_data[i]
      y_actual = y_data[i]
      y_guess = weight * x + bias

      weight_gradient += -(2 / n) * x * (y_actual - y_guess)
      bias_gradient += -(2 / n) * (y_actual - y_guess)
    
    weight -= (lr * weight_gradient)
    bias -= (lr * bias_gradient)

  session = SessionLocal()

  model_record = session.query(DBModel).filter(DBModel.id == model_id).first()

  if model_record:
    model_record.final_weight = weight
    model_record.final_bias = bias
    model_record.status = "Completed"
    session.commit()
  
  session.close()


@app.post("/train")
def start_training(data: TrainInput, background_tasks: BackgroundTasks):
  session = SessionLocal()

  new_model = DBModel(
    status = "Training",
    learning_rate = data.learning_rate,
    final_weight = 0.0,
    final_bias = 0.0
  )

  session.add(new_model)
  session.commit()

  session.refresh(new_model)
  model_id = new_model.id
  session.close()

  background_tasks.add_task(
    train_model_background,
    model_id = model_id,
    x_data = data.x_data,
    y_data = data.y_data,
    lr = data.learning_rate,
    iterations = data.iterations
  )

  return {
    "message": "Training job started successfully in the background.",
    "model_id": model_id,
    "status": "Training"
  }

@app.get("/model-status/{model_id}")
def get_model_status(model_id: int):
  session = SessionLocal()

  model_record = session.query(DBModel).filter(DBModel.id == model_id).first()
  session.close()

  if not model_record:
    return {"error": f"Model ID {model_id} not found."}
  
  return {
    "model_id": model_record.id,
    "status": model_record.status,
    "learning_rate": model_record.learning_rate,
    "final_weight": model_record.final_weight,
    "final_bias": model_record.final_bias
  }

class PredictInput(BaseModel):
  model_id: int
  x_value: float

@app.post("/predict")
def make_prediction(data: PredictInput):
  session = SessionLocal()

  model_record = session.query(DBModel).filter(DBModel.id == data.model_id).first()

  if not model_record:
    return {"error": f"Model ID {data.model_id} not found."}
  
  if model_record.status != "Completed":
    return {"error": f"Model {model_record.id} is currently {model_record.status}. Please wait."}
  
  prediction = model_record.final_weight * data.x_value + model_record.final_bias

  return {
    "model_used": model_record.id,
    "input_value": data.x_value,
    "prediction_value": prediction
  }
