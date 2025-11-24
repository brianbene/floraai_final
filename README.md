# FloraAI: The Field Guide That Grows With You

## Introduction & Problem Statement

**Punchy One Liner:** "The field guide that grows with you"

**Main problem:** Gardening will become ever more important as rising food prices will force individuals to start tending food for themselves, but gardening is also a difficult proposition

**Main goal:** Connecting affordability with intelligence: an integration of the power of AI so users have a reference guide with them as they go on their gardening journey

## Data Collection & Preparation

- Connected via API with USDA, NASA, OpenStreetMap
- Compared based on prompts set and connected to Gemini 3.1
- Further refined via zipcode and food based user inputs: able to create custom dataframe per user by zipcode by food item

## Exploratory Data Analysis (EDA)

**Data Collection Analysis:**
- Issues with finding accurate weather data due to government shutdown
- For our MVP, it was a simple webscrape for using icrawler. Only issues were access to some sites to allow for image download. For the final product, we need to ensure no privacy laws are violated.

## Feature Engineering

Three models were trained: a NB model, self-made CNN, and transfer learning on RESNET 50. For data preprocessing, the images were resized to accommodate the target model:

- **NB:** Images flattened to a 1D vector.
- **CNN:** Images resized to 128 for CNN. Images normalize for ease of processing
- **RESNET50:** resized to 224 and normalized to standard ImageNet values

## Model Selection & Evaluation

**Models:**
- NaiveBayes approach as a baseline.
- Simple CNN architecture-exploring simpler approaches prior to transfer learning
- RESNET50 offers a more robust approach

**Evaluation Metrics:**
- AUROC/Confusion Matrix were employed to assess the accuracy of the models.
- One Vs Rest AUROC used-multi class problem
- Confusion matrices to compare TPR, FPR

## Results & Interpretation

- MVP delivers value to users by giving results specific to their location
- Photo ID – able to upload image or use camera
- AI-powered guidance can help make gardening more accessible

## Deployment & Integration

- Expand proof of concept with initial user base
- Determine most viable revenue model
- Expand data sets to cater to larger global markets

## Contributors

- **Hyun:** Introduction & Problem Statement, Data Collection & Preparation, EDA
- **Brian:** EDA, Feature Engineering, Model Selection & Evaluation
- **Beth:** Results & Interpretation, Deployment & Integration
- **Mo:** Ethical & Societal Implications, Future Work & Recommendations, Conclusion & Reflection
