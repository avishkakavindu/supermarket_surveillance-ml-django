from django.urls import path, include
from api.api_views import *
from rest_framework.routers import DefaultRouter

router = DefaultRouter()
router.register('product', ProductViewSet, basename='product')

urlpatterns = [
    path('auth/', include('djoser.urls')),
    path('auth/', include('djoser.urls.jwt')),
    path('mask-predictions/', MaskDetectionAPIView.as_view(), name='mask_detection'),
    path('similar-products/<int:pk>/', SimilarProductAPIView.as_view(), name='similar_products'),
    path('shelf-detail/<int:pk>/', ShelfRetrieveAPIView.as_view(), name='shelf_detail'),
    path('facial-expresion-prediction/', FeedBackAPIView.as_view(), name='feed_back_predictions'),
    path('fraud-prediction/', FraudDetectionAPIView.as_view(), name='fraud_detection'),
    path('fraud-prediction/<int:incident_id>/', FraudDetectionAPIView.as_view(), name='fraud_detection'),
    path('crowd/', CrowdAPIView.as_view(), name='crowd'),
    path('association-rules/', AssociateRuleMiningDiscountAPIView.as_view(), name='assoc_rules'),
    path('discount/<int:customer_id>/', DiscountAPIView.as_view(), name='discount'),
    path('order/<int:customer_id>/', OrderAPIView.as_view(), name='order'),
    path('crowd-forecast/', FutureCrowdForecastAPIView.as_view(), name='crowd_forecast')
] + router.urls
