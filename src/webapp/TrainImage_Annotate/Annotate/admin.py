# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.contrib import admin

# Register your models here.
from Annotate.models import AnnotationData
from Annotate.models import PolygonCoordinates

admin.site.register(AnnotationData)
admin.site.register(PolygonCoordinates)