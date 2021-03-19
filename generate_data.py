from pycaret.classification import * # Preprocessing, modelling, interpretation, deployment...
import pandas as pd # Basic data manipulation
import dabl as db # Summary plot
from sklearn.model_selection import train_test_split # Data split
from sdv.tabular import CopulaGAN # Synthetic data
from sdv.evaluation import evaluate # Evaluate synthetic data
from btb.tuning import Tunable, GCPTuner # CopulaGAN optimising
from btb.tuning import hyperparams as hp  # Set hyperparameters for optimising
import joblib # Saving preparation steps

# Read and output the top 5 rows
original_data = pd.read_csv('KaggleV2-May-2016.csv')

# Split real data into training + test set
#extract features
def extract_features(dataset):
  # get month, day name and hour from Start Time after convert
  dataset['Appointment_year'] = dataset['AppointmentDay'].dt.year
  dataset['Appointment_month'] = dataset['AppointmentDay'].dt.month
  dataset['Appointment_day'] = dataset['AppointmentDay'].dt.day
  dataset['Appointment_day_name'] = dataset['AppointmentDay'].dt.day_name()
  #appointment hour is always 0 so we leave it out

  # get month and day name and hour from Start Time after convert
  dataset['Register_year'] = dataset['RegisterDay'].dt.year
  dataset['Register_month'] = dataset['RegisterDay'].dt.month
  dataset['Register_day'] = dataset['RegisterDay'].dt.day
  dataset['Register_day_name'] = dataset['RegisterDay'].dt.day_name()
  dataset['Register_hour'] = dataset['RegisterDay'].dt.hour
  dataset.drop('AppointmentDay', axis=1, inplace=True)
  dataset.drop('RegisterDay', axis=1, inplace=True)
def convert_datetime(dataset):
  dataset.rename(columns={'Handcap':'Handicap'},inplace=True)
  dataset.rename(columns={"ScheduledDay":"RegisterDay"},inplace=True)
  dataset['AppointmentDay'] = pd.to_datetime(dataset['AppointmentDay']).dt.tz_localize(None)
  dataset['RegisterDay'] = pd.to_datetime(dataset['RegisterDay']).dt.tz_localize(None)

convert_datetime(original_data)
tuner = GCPTuner(Tunable({
          'epochs': hp.IntHyperParam(min = 1, max = 2),
          'batch_size' : hp.IntHyperParam(min = 1, max = 100),
          'embedding_dim' : hp.IntHyperParam(min = 1, max = 100),
          'gen' : hp.IntHyperParam(min = 1, max = 1000),
          'dim_gen' : hp.IntHyperParam(min = 1, max = 1000)
        }))

best_score = float('-inf') # Keep track of best score
best_params = 0
real = original_data[original_data["No-show"] == "Yes"] # Filter to only those employees that left
# real = real.sample(n=500, replace=False, random_state=1)
## TRAINING LOOP START ##
model = None
for _ in range(10):
  print(_)
  # Get the hyperparameters for this loop
  proposal = None
  try:
    proposal = tuner.propose(1)
  except:
    print("error")
    continue
  # Create the CopulaGAN
  # NOTE - batch_size is multiplied by 10 as needs to be a factor of 10
  model = CopulaGAN(primary_key = "AppointmentID", 
                    embedding_dim = proposal['embedding_dim'],
                    generator_dim = (proposal['gen'], proposal['gen']),
                    discriminator_dim = (proposal['dim_gen'], proposal['dim_gen']),
                    batch_size = proposal['batch_size'] * 10,
                    epochs = proposal['epochs'])
  
  # Fit the CopulaGAN
  print("fit")
  model.fit(real)
  print("sample")
  # Create 40000 rows of data
  synth_data = model.sample(500, max_retries = 300)
  
  # Evaluate the synthetic data against the real data
  score = evaluate(synthetic_data = synth_data, real_data = real)
  print(score)
  # If the new hyperparameters beat the best ones, store them along with the score
  if score > best_score:
    best_params = proposal
    best_score = score

  # Record the hyperparameters and score      
  tuner.record(proposal, score)
  # except:
  #   print(f"error on tuner proposal {_}")
  

## TRAINING LOOP END ##


print('Best score obtained: ', best_score)
print('Best parameters: ', best_params)
