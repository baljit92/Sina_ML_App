from django.conf.urls import url

from . import views

urlpatterns = [
	url(r'^login/$', views.login, name='login'),
	url(r'^logout/$', views.logout, name='logout'),
	url(r'^homepage/$', views.home, name='home'),
	url(r'^checklogin/$', views.checklogin, name='checklogin'),
	url(r'^modelTest/$', views.modelTest, name='modelTest'),
	url(r'^modelTestFolder/$', views.modelTestFolder, name='modelTestFolder'),
	url(r'^imageEval/$', views.imgEval, name='evalImg'),

	url(r'^uploadType/(?P<type>\w{0,50})/$', views.uploadType, name='uploadType'),
	url(r'^normalImages/$', views.loadNormalImages, name='normalImages'),
	url(r'^abnormalImages/$', views.loadAbnormalImages, name='abnormalImages'),
	url(r'^annotationData/$', views.annotationData, name='annotationData'),
	url(r'^getNextPath/$', views.getNextPath, name='getNextPath'),
	url(r'^getPrevPath/$', views.getPrevPath, name='getPrevPath'),
	url(r'^getForwardPath/$', views.getForwardPath, name='getForwardPath'),
	url(r'^annotationDataPost/$', views.annotationDataPost, name='annotationDataPost'),
	url(r'^annotationExport/$', views.exportAnnotation, name='annotationExport'),
	url(r'^updateImageOrder/$', views.updateImageOrder, name='updateImageOrder'),
	url(r'^loadAllImagesDB/$', views.loadAllImagesDB, name='loadAllImagesDB'),
	url(r'^saveUser/$', views.saveUser, name='saveUser')

]
