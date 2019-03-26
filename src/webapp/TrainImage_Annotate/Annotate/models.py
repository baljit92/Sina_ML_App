from __future__ import unicode_literals
from django.db import models

# from auliv_app.models.FileStorage import FileStorage
# from auliv_app.models.UserEmoticons import UserEmoticons
# from auliv_app.views.user.profile.ProfileManagementSerializer import ProfileManagementSerializer, AppUserSerializer

class PolygonCoordinates(models.Model):
    cords = models.TextField(null=True, blank=True)
    username = models.CharField(max_length=25, null=True, blank=True)

    def __str__(self):
        return str(self.id) + " " + str(self.username) + " " + str(self.cords)

    class Meta:
        verbose_name_plural = 'Polygon Coordinates'

class AnnotationData(models.Model):

    """
    AnnotationData model is used for annotation details
    """
    # date, file name, annotation_result(boolean - normal/abnormal/not clear)
    username = models.CharField(max_length=15, null=True, blank=True)
    date = models.DateTimeField(auto_now_add=True)
    file_name = models.CharField(max_length=200, blank=True)
    poly_cords = models.ManyToManyField(PolygonCoordinates)
    annotation_result = models.IntegerField(default=7)
    # comment = models.CharField(max_length=255, blank=True, default="")

    def __str__(self):
        """
        :return:

        """
        return str(self.id) + " " + str(self.username) + " " + str(self.poly_cords)

    class Meta:
        verbose_name_plural = 'Annotation Data'


class ImageModel(models.Model):
    type_folder= models.CharField(max_length=15, null=True, blank=True)
    batch_number = models.CharField(max_length=15, null=True, blank=True)
    ground_truth = models.IntegerField(default=3)
    model_truth = models.IntegerField(default=3)
    model_max = models.FloatField(default=0.0)
    model_min = models.FloatField(default=0.0)
    model_avg = models.FloatField(default=0.0)
    file_name = models.CharField(max_length=500, null=True, blank=True)
    image_type = models.IntegerField(default=0)
    reviewer1 = models.IntegerField(blank=True, null=True)
    reviewer2 = models.IntegerField(blank=True, null=True)
    reviewer3 = models.IntegerField(blank=True, null=True)
    reviewer4 = models.IntegerField(blank=True, null=True)
    reviewer5 = models.IntegerField(blank=True, null=True)


    class Meta:
        verbose_name_plural = 'Image Model'

class ImageOrder(models.Model):
    img_id = models.IntegerField(blank=True, null=True)
    reviewer1 = models.IntegerField(blank=True, null=True)
    reviewer2 = models.IntegerField(blank=True, null=True)
    reviewer3 = models.IntegerField(blank=True, null=True)
    reviewer4 = models.IntegerField(blank=True, null=True)
    reviewer5 = models.IntegerField(blank=True, null=True)


    class Meta:
        verbose_name_plural = 'Image Order'

class UserModel(models.Model):
    username = models.CharField(max_length=15, null=True, blank=True)
    img_array = models.TextField(blank=True)

    def set_imgarray(self, x):
        self.img_array = json.dumps(x)

    def get_imgarray(self):
        return json.loads(self.img_array)

    class Meta:
        verbose_name_plural = 'User Model'