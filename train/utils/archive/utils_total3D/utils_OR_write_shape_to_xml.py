import numpy as np
import xml.etree.ElementTree as et
from xml.dom import minidom

def transformToXml(root ):
    rstring = et.tostring(root, 'utf-8')
    pstring = minidom.parseString(rstring)
    xmlString = pstring.toprettyxml(indent="    ")
    xmlString= xmlString.split('\n')
    xmlString = [x for x in xmlString if len(x.strip()) != 0 ]
    xmlString = '\n'.join(xmlString )
    return xmlString

def addMaterial_diffuse(root, name, albedo_rough_list, uvScaleValue = None, partId=0):
    for mat in albedo_rough_list:
        bsdf = et.SubElement(root, 'bsdf' )
        bsdf.set('type', 'microfacet')
        bsdf.set('id', name + '_' + str(partId) )
        # matId = matName.split('/')[-1]
        # matFile = osp.join(adobeRootAbs, matId, 'mat.txt')

        albedo_list, rough_value = mat[0], mat[1]

        bsdf.set('type', 'microfacet')
        # Add uv scale
        if uvScaleValue is not None:
            uvScale = et.SubElement(bsdf, 'float')
            uvScale.set('name', 'uvScale')
            uvScale.set('value', '%.3f' % uvScaleValue )

        # Add new albedo
        albedo = et.SubElement(bsdf, 'rgb' )
        albedo.set('name', 'albedo' )
        albedo.set('value', ' '.join([str(x) for x in albedo_list]) )

        # Add albedo scale
        albedoScale = et.SubElement(bsdf, 'rgb')
        albedoScale.set('name', 'albedoScale')
        # albedoScaleValue = np.random.random(3) * 0.6 + 0.7
        albedoScaleValue = [1., 1., 1.]
        albedoScale.set('value', '%.3f %.3f %.3f' %
                (albedoScaleValue[0], albedoScaleValue[1], albedoScaleValue[2] ) )

    #     # Add new normal
    #     normal = et.SubElement(bsdf, 'texture' )
    #     normal.set('name', 'normal')
    #     normal.set('value', ' '.join([str(x) for x in normal_list]) )
    #    # normal.set('type', 'bitmap')
    #     # normalfile = et.SubElement(normal, 'string')
    #     # normalfile.set('name', 'filename')
    #     # normalfile.set('value', osp.join(matName, 'tiled', 'normal_tiled.png') )

        # Add new roughness
        roughness = et.SubElement(bsdf, 'float' )
        roughness.set('name', 'roughness')
        # roughness.set('type', 'bitmap')
        roughness.set('value', str(rough_value))
        # roughnessfile = et.SubElement(roughness, 'string')
        # roughnessfile.set('name', 'filename')
        # roughnessfile.set('value', osp.join(matName, 'tiled', 'rough_tiled.png') )

        # Add roughness scale
        roughScale = et.SubElement(bsdf, 'float')
        roughScale.set('name', 'roughnessScale')
        # roughScaleValue = np.random.random() * 1.0 + 0.5
        roughScaleValue = 1.
        roughScale.set('value', '%.3f' % roughScaleValue  )

    return root

def addShape(root, name, fileName, transforms = None, materials = None, scaleValue = None):
    shape = et.SubElement(root, 'shape')
    shape.set('id', '%s_object' % name )
    shape.set('type', 'obj' )

    stringF = et.SubElement(shape, 'string' )
    stringF.set('name', 'filename' )
    stringF.set('value', fileName )

    if not scaleValue is None:
        transform = et.SubElement(shape, 'transform')
        transform.set('name', 'toWorld')
        scale = et.SubElement(transform, 'scale')
        scale.set('value', '%.5f' % scaleValue )


    if transforms != None:
        transform = et.SubElement(shape, 'transform')
        transform.set('name', 'toWorld')
        for tr in transforms:
            if tr[0] == 's':
                s = tr[1]
                scale = et.SubElement(transform, 'scale')
                scale.set('x', '%.6f' % s[0] )
                scale.set('y', '%.6f' % s[1] )
                scale.set('z', '%.6f' % s[2] )

            elif tr[0] == 'rot':
                rotMat = tr[1]
                rotTr = rotMat[0,0] + rotMat[1,1] + rotMat[2,2]
                rotCos = np.clip((rotTr - 1) * 0.5, -1, 1 )
                rotAngle = np.arccos(np.clip(rotCos, -1, 1 ) )
                if np.abs(rotAngle) > 1e-3 and np.abs(rotAngle - np.pi) > 1e-3:
                    rotSin = np.sqrt(1 - rotCos * rotCos )
                    rotAxis_x = 0.5 / rotSin * (rotMat[2, 1] - rotMat[1, 2] )
                    rotAxis_y = 0.5 / rotSin * (rotMat[0, 2] - rotMat[2, 0] )
                    rotAxis_z = 0.5 / rotSin * (rotMat[1, 0] - rotMat[0, 1] )

                    norm = rotAxis_x * rotAxis_x \
                            + rotAxis_y * rotAxis_y \
                            + rotAxis_z * rotAxis_z
                    norm = np.sqrt(norm )

                    rotate = et.SubElement(transform, 'rotate')
                    rotate.set('x', '%.6f' % (rotAxis_x / norm ) )
                    rotate.set('y', '%.6f' % (rotAxis_y / norm ) )
                    rotate.set('z', '%.6f' % (rotAxis_z / norm ) )
                    rotate.set('angle', '%.6f' % (rotAngle / np.pi * 180 ) )

            elif tr[0] == 't':
                t = tr[1]
                trans = et.SubElement(transform, 'translate')
                trans.set('x', '%.6f' % t[0] )
                trans.set('y', '%.6f' % t[1] )
                trans.set('z', '%.6f' % t[2] )
            else:
                print('Wrong: unrecognizable type of transformation!' )
                assert(False )

    if materials is not None:
        for mat in materials:
            matName, partId = mat[1], mat[0]
            bsdf = et.SubElement(shape, 'ref' )
            bsdf.set('name', 'bsdf')
            bsdf.set('id', name + '_' + str(partId) )
    return root


def addAreaLight(root, name, fileName, transforms = None, rgbColor=None):
    shape = et.SubElement(root, 'shape')
    shape.set('id', '%s_object' % name )
    shape.set('type', 'obj' )

    stringF = et.SubElement(shape, 'string' )
    stringF.set('name', 'filename' )
    stringF.set('value', fileName )

    emitter = et.SubElement(shape, 'emitter')
    emitter.set('type', 'area')

    
    if rgbColor is None:
        # rgbColor = sampleRadianceFromTemp()
        assert False
    rgb = et.SubElement(emitter, 'rgb')
    rgb.set('value', '%.3f %.3f %.3f' % (rgbColor[0], rgbColor[1], rgbColor[2] ) )

    if transforms != None:
        transform = et.SubElement(shape, 'transform')
        transform.set('name', 'toWorld')
        for tr in transforms:
            if tr[0] == 's':
                s = tr[1]
                scale = et.SubElement(transform, 'scale' )
                scale.set('x', '%.6f' % s[0] )
                scale.set('y', '%.6f' % s[1] )
                scale.set('z', '%.6f' % s[2] )

            elif tr[0] == 'rot':
                rotMat = tr[1]
                rotTr = rotMat[0,0] + rotMat[1,1] + rotMat[2,2]
                rotCos = (rotTr - 1) * 0.5
                rotAngle = np.arccos(np.clip(rotCos, -1, 1 ) )
                if np.abs(rotAngle) > 1e-2:
                    rotSin = np.sqrt(1 - rotCos * rotCos )
                    rotAxis_x = 0.5 / rotSin * (rotMat[2, 1] - rotMat[1, 2] )
                    rotAxis_y = 0.5 / rotSin * (rotMat[0, 2] - rotMat[2, 0] )
                    rotAxis_z = 0.5 / rotSin * (rotMat[1, 0] - rotMat[0, 1] )

                    norm = rotAxis_x * rotAxis_x \
                            + rotAxis_y * rotAxis_y \
                            + rotAxis_z * rotAxis_z
                    norm = np.sqrt(norm )

                    rotate = et.SubElement(transform, 'rotate' )
                    rotate.set('x', '%.6f' % (rotAxis_x / norm ) )
                    rotate.set('y', '%.6f' % (rotAxis_y / norm ) )
                    rotate.set('z', '%.6f' % (rotAxis_z / norm ) )
                    rotate.set('angle', '%.6f' % (rotAngle / np.pi * 180 ) )

            elif tr[0] == 't':
                t = tr[1]
                trans = et.SubElement(transform, 'translate')
                trans.set('x', '%.6f' % t[0] )
                trans.set('y', '%.6f' % t[1] )
                trans.set('z', '%.6f' % t[2] )
            else:
                print('Wrong: unrecognizable type of transformation!' )
                assert(False )
    return root
