import re
from django.core.files.storage import default_storage
from rest_framework.views import APIView
from rest_framework import status, viewsets, generics
from django.shortcuts import get_object_or_404
from rest_framework.response import Response
from tensorflow.keras.models import load_model
from api.models import *
from api.serializers import *
import cv2
import numpy as np
import pandas as pd
import dlib
from datetime import datetime, timedelta
from django.db.models import Sum
from rest_framework_simplejwt.authentication import JWTTokenUserAuthentication
from rest_framework import authentication, permissions


def process_image(image, size=224):
    # preprocess image
    img = cv2.imread(image)
    img = cv2.resize(img, dsize=(size, size), interpolation=cv2.INTER_AREA)

    img = img.reshape((1, size, size, 3)).astype(np.float32)

    return img / 255


class MaskDetectionAPIView(APIView):
    """ Detect masks, record if incident detected """

    def post(self, request, *args, **kwargs):
        uploaded_image = request.FILES['image']

        model_path = 'api/trained_models/mobilenet_facemask_12k_model.h5'

        file_name = default_storage.save('image.png', uploaded_image)
        # get path
        file_path = default_storage.path(file_name)

        image = process_image(file_path)
        # load trained model
        model = load_model(model_path)
        # get predictions
        pred = model.predict(image)
        # best prediction indexes inversely sorted
        best = (-pred).argsort()[0]
        # predictions to text
        labels = ['with_mask', 'without_mask']
        sorted_preds = [labels[i] for i in best]

        if sorted_preds != labels[0]: # without mask
            obj = Incident.objects.create(type=Incident.MASK, media_file=uploaded_image)
            alert = True
            detail = "Not wearing a mask!"
        else:
            alert = False
            detail = "Wearing a mask!"

        context = {
            'alert': alert,
            'predictions': sorted_preds[0],
            'detail': detail
        }

        return Response(context, status=status.HTTP_200_OK)


class SimilarProductAPIView(APIView):
    """ Suggest similar products based on description """

    def remove_html_elements(self, text):
        return re.sub(r"<[a-z/]+>", " ", text)

    def remove_special_char(self, text):
        return re.sub(r"[^A-Za-z]+", " ", text)

    def remove_whitespaces(self, text):
        return re.sub(' +', ' ', text)

    def get_df(self):
        descriptions = list(Product.objects.all().values_list('id', 'description'))

        import pandas as pd

        df = pd.DataFrame(descriptions, columns=['id', 'description'])
        return df

    def get_descriptions_similarity_score(self):
        df = self.get_df()
        df['description'] = df['description'].apply(lambda x: self.remove_html_elements(x))
        df['description'] = df['description'].apply(lambda x: self.remove_special_char(x))
        df['description'] = df['description'].apply(lambda x: self.remove_whitespaces(x))
        df['description'] = df['description'].apply(lambda x: x.lower())

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import linear_kernel

        t_vec = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 3),
            min_df=0,
            stop_words='english')

        tvec_matrix = t_vec.fit_transform(df['description'])
        cosine_similarities = linear_kernel(tvec_matrix, tvec_matrix)

        results = {}

        for idx, row in df.iterrows():
            # print('{} {}'.format(idx, row))
            similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
            similar_items = [(cosine_similarities[idx][i], df['id'][i]) for i in similar_indices]

            results[row['id']] = similar_items

        return results

    # retrieve item description
    def get_item(self, id):
        df = self.get_df()
        description = df.loc[df['id'] == id]['description'].tolist()[0]
        return description

    def get_recommendations(self, product_id, num_of_rec=3):
        # get n number of records (n = num_of_rec)
        results = self.get_descriptions_similarity_score()
        recs = results[product_id][:num_of_rec]
        ids = [rec[1] for rec in recs]

        return ids

    def get(self, request, pk, *args, **kwargs):
        product_id = pk

        recommendations = self.get_recommendations(product_id)

        products = Product.objects.filter(id__in=recommendations)

        serializer = ProductSerializer(products, many=True)

        context = {
            'products': serializer.data
        }

        return Response(context, status=status.HTTP_200_OK)


class ProductViewSet(viewsets.ModelViewSet):
    """ Product  based viewsets, get, post, create """

    queryset = Product.objects.all()
    serializer_class = ProductSerializer

    def list(self, request, *args, **kwargs):
        serializer_class = ProductSerializer(self.queryset, many=True)
        return Response(serializer_class.data, status=status.HTTP_200_OK)

    def retrieve(self, request, pk=None, *args, **kwargs):
        product = get_object_or_404(self.queryset, pk=pk)
        serializer_class = ProductSerializer(product)
        return Response(serializer_class.data, status=status.HTTP_200_OK)


class ShelfRetrieveAPIView(generics.RetrieveAPIView):
    """ Get shelf data by shelf id """

    queryset = Shelf.objects.all()
    serializer_class = ShelfSerializer


class FeedBackAPIView(APIView):
    """ Facial expression recognition """

    def post(self, request, *args, **kwargs):
        uploaded_image = request.FILES['image']
        camera_id = request.POST['camera_id']

        model_path = 'api/trained_models/MobileNet_facialExpression.h5'

        file_name = default_storage.save('image.png', uploaded_image)
        # get path
        file_path = default_storage.path(file_name)

        image = process_image(file_path)
        # load trained model
        model = load_model(model_path)
        # get predictions
        pred = model.predict(image)
        # best prediction indexes inversely sorted
        best = (-pred).argsort()[0]
        # predictions to text
        labels = ['anger', 'fear', 'happy', 'sadness', 'surprise']
        sorted_preds = [labels[i] for i in best]

        obj = FeedBack.objects.create(
            camera=Camera.objects.get(pk=camera_id),
            top_3_feedback=', '.join(sorted_preds[:3])
        )

        serializer = FeedBackSerializer(obj)

        context = {
            'detail': serializer.data
        }

        return Response(context, status=status.HTTP_200_OK)


class FraudDetectionAPIView(APIView):
    """ 'Burglary', 'NormalVideos', 'Robbery', 'Shoplifting', 'Stealing' detection """

    def get(self, request, *args, **kwargs):
        incident_id = kwargs.get('incident_id')

        if incident_id is None:
            incidents = Incident.objects.all()
            serializer = IncidentSerializer(incidents, many=True)

        else:
            incident = Incident.objects.get(id=incident_id)
            serializer = IncidentSerializer(incident)

        context = {
            'detail': serializer.data
        }

        return Response(context, status=status.HTTP_200_OK)

    def post(self, request, *args, **kwargs):
        uploaded_image = request.FILES['image']
        camera_id = request.POST['camera_id']

        model_path = 'api/trained_models/CrimeRecognitionDenseNet.h5'

        file_name = default_storage.save('image.png', uploaded_image)
        # get path
        file_path = default_storage.path(file_name)

        image = process_image(file_path, 64)
        # load trained model
        model = load_model(model_path)
        # get predictions
        pred = model.predict(image)
        # best prediction indexes inversely sorted
        best = (-pred).argsort()[0]
        # predictions to text
        labels = ['Robbery', 'Stealing', 'Burglary', 'NormalVideos', 'Shoplifting']
        sorted_preds = [labels[i] for i in best]

        if sorted_preds[0] != 'NormalVideos':
            obj = Incident.objects.create(
                type=Incident.FRAUD,
                media_file=uploaded_image,
                camera_id=Camera.objects.get(pk=camera_id)
            )
            serializer = IncidentSerializer(obj)
            incident = True

            context = {
                'incident': incident,
                'top_3_predictions': sorted_preds[:3],
                'detail': serializer.data
            }

        else:
            incident = False
            context = {
                'incident': incident,
                'top_3_predictions': None,
                'detail': 'No Incident Detected!',
                'preds': sorted_preds
            }

        return Response(context, status=status.HTTP_200_OK)


class CrowdAPIView(APIView):
    """ Records retrieve crowd inside the super market """

    def detect_faces(self, frame):
        detector = dlib.get_frontal_face_detector()
        # gray scaling
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect faces
        faces = detector(gray)

        return len(faces), faces

    def draw_rectangles(self, frame, faces):
        for i, face in enumerate(faces):
            x, y = face.left(), face.top()
            x1, y1 = face.right(), face.bottom()
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

            # Increment the iterartor each time you get the coordinates
            i = i + 1

            # Adding face number to the box detecting faces
            cv2.putText(frame, 'face num' + str(i), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # print(face, i)
        return frame

    def get(self, request, *args, **kwargs):
        from datetime import timedelta
        from django.utils import timezone

        try:
            now_data = Crowd.objects.filter().latest('date_time')
        except Crowd.DoesNotExist:
            context = {
                'detail': 'No crowd record found!'
            }

            return Response(context, status=status.HTTP_404_NOT_FOUND)

        last_year = datetime.today() - timedelta(days=365)
        last_month = datetime.today() - timedelta(days=30)
        last_week = datetime.today() - timedelta(days=7)

        year_data = Crowd.objects.filter(date_time__gte=last_year)
        month_data = Crowd.objects.filter(date_time__gte=last_month)
        week_data = Crowd.objects.filter(date_time__gte=last_week)

        crowd = {
            'now': now_data,
            'week': week_data,
            'month': month_data
        }

        context = {
            'crowd_now': CrowdSerializer(crowd['now']).data,
            'crowd_week': CrowdSerializer(crowd['week'], many=True).data,
            'crowd_month': CrowdSerializer(crowd['month'], many=True).data,
            'crowd_total_week': week_data.aggregate(Sum('crowd_count'))['crowd_count__sum'],
            'crowd_total_month': month_data.aggregate(Sum('crowd_count'))['crowd_count__sum'],
            'crowd_total_year': year_data.aggregate(Sum('crowd_count'))['crowd_count__sum'],
        }

        return Response(context, status=status.HTTP_200_OK)

    def post(self, request, *args, **kwargs):
        uploaded_image = request.FILES['image']

        file_name = default_storage.save('image.png', uploaded_image)
        # get path
        file_path = default_storage.path(file_name)

        frame = cv2.imread('imgs/faces.png')

        num_of_faces, faces = self.detect_faces(frame)

        frame = self.draw_rectangles(frame, faces)

        obj = Crowd.objects.create(
            crowd_count=num_of_faces,
            media_file=frame
        )

        serializer = CrowdSerializer(obj)

        context = {
            'detail': serializer.data

        }

        return Response(context, status=status.HTTP_200_OK)


class AssociateRuleMiningDiscountAPIView(APIView):
    """ Gets associate rules and allow to assign discounts  """

    # def get_df(self):
    #     orders = list(Order.objects.all().values_list('id', 'description'))
    #
    #     import pandas as pd
    #
    #     df = pd.DataFrame(descriptions, columns=['id', 'description'])
    #     return df

    def get_association_rules(self):
        orders = OrderedProduct.objects.all().values('order__id', 'product__name')

        df = pd.DataFrame.from_records(orders)
        data = df.groupby(['order__id'])['product__name'].apply(list).to_list()

        from mlxtend.preprocessing import TransactionEncoder

        te = TransactionEncoder()
        te_ary = te.fit(data).transform(data)
        transactions = pd.DataFrame(te_ary, columns=te.columns_)

        count = transactions.loc[:, :].sum()
        pop_item = count.sort_values(0, ascending=False).head(40)
        pop_item = pop_item.to_frame()
        pop_item = pop_item.reset_index()
        pop_item = pop_item.rename(columns={"index": "items", 0: "count"})

        from mlxtend.frequent_patterns import apriori, association_rules

        frequent_items = apriori(transactions, min_support=0.6, use_colnames=True, verbose=1)
        # add length of the itemset
        frequent_items['length'] = frequent_items['itemsets'].apply(lambda x: len(x))
        # frequent_items = frequent_items.loc[frequent_items['length'] >= 2]

        rules = association_rules(
            frequent_items,
            metric="confidence",
            min_threshold=0.5
        )

        selected_details = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

        return pop_item, selected_details   # pop items = popular items

    def get(self, request, *args, **kwargs):
        pop_item, selected_details = self.get_association_rules()

        context = {
            'product_selling_counts ': pop_item,
            'detail': selected_details
        }

        return Response(context, status=status.HTTP_200_OK)


class DiscountAPIView(APIView):
    """ Calculates the discount based on association rules """

    authentication_classes = [JWTTokenUserAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def combinations(self, items):
        from itertools import compress, product

        return list(list(set(compress(items, mask))) for mask in product(*[[0, 1]] * len(items)))

    def get(self, request, *args, **kwargs):
        # get latest order
        order = Order.objects.filter(user=request.user.id).latest('date_time')

        products_in_order = order.product.values('id').values_list('name', flat=True)

        # association rules
        armav = AssociateRuleMiningDiscountAPIView()
        pop_item, association_rules = armav.get_association_rules()

        association_rules['antecedents'] = association_rules['antecedents'].apply(list)

        selected = pd.DataFrame(data=None, columns=association_rules.columns, index=association_rules.index)
        selected.drop(selected.index, inplace=True)

        all_combinations = self.combinations(products_in_order)

        for idx, row in association_rules.iterrows():
            all_present = True
            for item in row['antecedents']:
                if item not in products_in_order:
                    all_present = False
                    break
            if all_present:
                for item in row['consequents']:
                    if item not in products_in_order:
                        all_present = False
                        break

            if all_present:
                selected.loc[-1] = row
                selected.index = selected.index + 1  # shifting index
                selected = selected.sort_index()  # sorting by index

        # get items that can give discount
        selected['consequents'] = selected['consequents'].apply(list)
        discount_items = [item for items in selected['consequents'].tolist() for item in items]

        products = Product.objects.filter(name__in=discount_items)

        serializer = ProductDiscountSerializer(products, many=True)

        context = {
            'product_with_discounts': serializer.data
        }

        return Response(context, status=status.HTTP_200_OK)


class OrderAPIView(APIView):
    """ Add products, retrieve order details """

    authentication_classes = [JWTTokenUserAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, *args, **kwargs):
        order = Order.objects.filter(user=request.user.id).latest('date_time')

        serializer = OrderSerializer(order)

        context = {
            'detail': serializer.data
        }

        return Response(context, status=status.HTTP_200_OK)

    def post(self, request, *args, **kwargs):
        product_id = request.POST['id']
        quantity = request.POST['quantity']

        order = Order.objects.filter(user=request.user.id).latest('date_time')

        try:
            product = Product.objects.get(pk=product_id)
        except Product.DoesNotExist:
            return Response({'detail': 'Product not found!'}, status=status.HTTP_404_NOT_FOUND)

        ordered_product, created = OrderedProduct.objects.update_or_create(
            order=order,
            product=product
        )

        from django.db.models import F

        ordered_product.quantity = F('quantity') + quantity
        ordered_product.save()

        loyalty, created = Loyalty.objects.update_or_create(
            user=User.objects.get(pk=request.user.id)
        )

        loyalty.points = F('points') + product.loyalty_points
        loyalty.save()

        serializer = OrderedProductSerializer(OrderedProduct.objects.get(pk=ordered_product.id))

        print(ordered_product)

        context = {
            'detail': 'Product added to order!',
            'data': serializer.data
        }

        return Response(context, status=status.HTTP_200_OK)