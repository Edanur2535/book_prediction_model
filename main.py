import pandas as pd
from surprise import Dataset,SVD,Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

ratings=pd.read_csv('ratings.csv')
ratings=ratings.rename(columns={'User-ID':'userID','Book-Rating':'rating','ISBN':'itemID'})
ratings=ratings[ratings['rating']>0]
reader=Reader(rating_scale=(0,10))
data=Dataset.load_from_df(ratings[['userID','itemID','rating']],reader)
train,test=train_test_split(data,test_size=0.2,random_state=42)
print(train.n_items)

model=SVD()
model.fit(train)

prediction=model.test(test)
rmse=accuracy.rmse(prediction)

user_id='12345'
rated_item=ratings[ratings['userID']==user_id]['itemID'].to_list()

all_rated_item=ratings['itemID'].unique()

unrated_item=[item for item in all_rated_item if item not in rated_item]

pred=[]
for item_id in unrated_item:
    predi=model.predict(user_id,item_id)
    pred.append((item_id,predi.est))

top_b=sorted(pred,key=lambda x:x[1],reverse=True)[:10]
for item_id,rating in top_b:
    print(f"Book:{item_id} - Prediction Of Rating:{round(rating,2)}")