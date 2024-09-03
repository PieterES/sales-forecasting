import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ah_package.training.utils import convert_log_to_units, load_model

# Initialize FastAPI app
app = FastAPI()

model = load_model()


# Define the prediction input structure
class PredictionInput(BaseModel):
    """The necessary inputs needed for the model to make a prediction

    Args:
        BaseModel: The input data
    """

    StoreCount: int
    ShelfCapacity: float
    PromoShelfCapacity: float
    IsPromo: bool
    ItemNumber: int
    CategoryCode: int
    GroupCode: int
    month: int
    weekday: int
    UnitSales_7: float = Field(..., alias="UnitSales_-7")
    UnitSales_14: float = Field(..., alias="UnitSales_-14")
    UnitSales_21: float = Field(..., alias="UnitSales_-21")

    class Config:
        allow_population_by_field_name = True


@app.post("/predict")
async def predict(input_data: list[float]):
    """The prediction function, maps the input to the correct format and turns it into a dictionary that can be used by the model to predict.

    Args:
        input_data (list[float]): The input data given by the user to estimate the predicted units.

    Raises:
        HTTPException: Exception raised if a value is missing.
        HTTPException: Exception raised if the model failed to predict.

    Returns:
        int: The actual predicted amount of units
    """
    if len(input_data) != 12:
        raise HTTPException(
            status_code=400,
            detail="Invalid input data. Exactly 12 values are required.",
        )

    # Map the list values to the corresponding fields
    mapped_data = {
        "StoreCount": input_data[0],
        "ShelfCapacity": input_data[1],
        "PromoShelfCapacity": input_data[2],
        "IsPromo": input_data[3],
        "ItemNumber": input_data[4],
        "CategoryCode": input_data[5],
        "GroupCode": input_data[6],
        "month": input_data[7],
        "weekday": input_data[8],
        "UnitSales_-7": input_data[9],
        "UnitSales_-14": input_data[10],
        "UnitSales_-21": input_data[11],
    }

    # Convert the mapped data into the correct format
    input_model = PredictionInput(**mapped_data)
    df = pd.DataFrame([input_model.dict(by_alias=True)])

    # Predict using the loaded model
    try:
        prediction = model.predict(df)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Model prediction failed: {str(e)}"
        )

    predicted_units = convert_log_to_units(prediction[0])

    return {"predicted_units": predicted_units}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
