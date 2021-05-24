from util import *

def tex2map(tex):
    albedo = ((tex[:,0:3,:,:].clamp(-1,1) + 1) / 2) ** 2.2

    normal_x  = tex[:,3,:,:].clamp(-1,1)
    normal_y  = tex[:,4,:,:].clamp(-1,1)
    normal_xy = (normal_x**2 + normal_y**2).clamp(min=0, max=1-eps)
    normal_z  = (1 - normal_xy).sqrt()
    normal    = th.stack((normal_x, normal_y, normal_z), 1)
    normal    = normal.div(normal.norm(2.0, 1, keepdim=True))

    rough = ((tex[:,5,:,:].clamp(-0.3,1) + 1) / 2) ** 2.2
    rough = rough.clamp(min=eps).unsqueeze(1).expand(-1,3,-1,-1)

    if tex.shape[1] == 9:
        specular = ((tex[:,6:9,:,:].clamp(-1,1) + 1) / 2) ** 2.2
        return albedo, normal, rough, specular

    return albedo, normal, rough

class Microfacet:
    def __init__(self, res, size, f0=0.04):
        self.res = res
        self.size = size
        self.f0 = f0
        self.eps = 1e-6

        self.initGeometry()

    def initGeometry(self):
        tmp = th.arange(self.res, dtype=th.float32).cuda()
        tmp = ((tmp + 0.5) / self.res - 0.5) * self.size
        y, x = th.meshgrid((tmp, tmp))
        self.pos = th.stack((x, -y, th.zeros_like(x)), 2)
        self.pos_norm = self.pos.norm(2.0, 2, keepdim=True)

    def GGX(self, cos_h, alpha):
        c2 = cos_h ** 2
        a2 = alpha ** 2
        den = c2 * a2 + (1 - c2)
        return a2 / (np.pi * den**2 + self.eps)

    def Beckmann(self, cos_h, alpha):
        c2 = cos_h ** 2
        t2 = (1 - c2) / c2
        a2 = alpha ** 2
        return th.exp(-t2 / a2) / (np.pi * a2 * c2 ** 2)

    def Fresnel(self, cos, f0):
        return f0 + (1 - f0) * (1 - cos)**5

    def Fresnel_S(self, cos, specular):
        sphg = th.pow(2.0, ((-5.55473 * cos) - 6.98316) * cos);
        return specular + (1.0 - specular) * sphg

    def Smith(self, n_dot_v, n_dot_l, alpha):
        def _G1(cos, k):
            return cos / (cos * (1.0 - k) + k)

        k = alpha * 0.5 + self.eps
        return _G1(n_dot_v, k) * _G1(n_dot_l, k)

    def normalize(self, vec):
        assert(vec.size(0)==self.N)
        assert(vec.size(1)==3)
        assert(vec.size(2)==self.res)
        assert(vec.size(3)==self.res)

        vec = vec / (vec.norm(2.0, 1, keepdim=True))
        return vec

    def getDir(self, pos):
        pos = th.from_numpy(pos).cuda()
        vec = (pos - self.pos).permute(2,0,1).unsqueeze(0).expand(self.N,-1,-1,-1)
        return self.normalize(vec), (vec**2).sum(1, keepdim=True).expand(-1,3,-1,-1)

    def AdotB(self, a, b):
        ab = (a*b).sum(1, keepdim=True).clamp(min=0).expand(-1,3,-1,-1)
        return ab

    def eval(self, textures, lightPos, cameraPos, light):
        self.N = textures.size(0)
        isSpecular = False
        if textures.size(1) == 9:
            isSpecular = True
            # print('Render Specular\n')

        if isSpecular:
            albedo, normal, rough, specular = tex2map(textures)
        else:
            albedo, normal, rough = tex2map(textures)
        light = light.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(albedo)

        v, _ = self.getDir(cameraPos)
        l, dist_l_sq = self.getDir(lightPos)
        h = self.normalize(l + v)

        n_dot_v = self.AdotB(normal, v)
        n_dot_l = self.AdotB(normal, l)
        n_dot_h = self.AdotB(normal, h)
        v_dot_h = self.AdotB(v, h)

        geom = n_dot_l / dist_l_sq

        D = self.GGX(n_dot_h, rough**2)
        # D = self.Beckmann(n_dot_h, rough**2)
        if isSpecular:
            F = self.Fresnel_S(v_dot_h, specular)
        else:
            F = self.Fresnel(v_dot_h, self.f0)
        G = self.Smith(n_dot_v, n_dot_l, rough**2)

        # lambert brdf
        f1 = albedo / np.pi
        if isSpecular:
            f1 *= (1 - specular)
        # cook-torrence brdf
        f2 = D * F * G / (4 * n_dot_v * n_dot_l + self.eps)

        # brdf
        kd = 1; ks = 1
        f = kd * f1 + ks * f2

        # rendering
        img = f * geom * light

        return img.clamp(0,1)

class MicrofacetPlaneCrop:
    #def __init__(self, res, size, bbox, planeNormal, f0=0.04):
    def __init__(self, res, bbox, planeNormal, imWidth = 160, imHeight = 120, fov=57, F0=0.05, cameraPos = [0, 0, 0], 
            envWidth = 16, envHeight = 8, isCuda = True):
        self.imHeight = imHeight
        self.imWidth = imWidth
        self.envWidth = envWidth
        self.envHeight = envHeight

        self.fov = fov/180.0 * np.pi
        self.F0 = F0
        #self.cameraPos = th.from_numpy(np.array(cameraPos, dtype=np.float32).reshape([1, 3, 1, 1]) )
        self.cameraPos = th.from_numpy(np.array(cameraPos, dtype=np.float32))
        self.isCuda = isCuda
        self.bbox = bbox # [relC, relR, relW, relH]
        self.planeNormal = np.array(planeNormal)

        #######
        self.res = res # resolution of texture
        #self.size = size # actual size of texture
        #self.f0 = f0
        self.eps = 1e-6

        self.initGeometry()
        self.Rot = self.getRot(th.tensor([0.0, 0.0, 1.0]), th.from_numpy(self.planeNormal.astype(np.float32)) ).cuda()
        self.initEnv()
        self.up = th.Tensor([0,1,0] )

        if isCuda:
            self.v = self.v.cuda()
            self.pos = self.pos.cuda()
            self.up = self.up.cuda()
            self.ls = self.ls.cuda()
            self.envWeight = self.envWeight.cuda()

    def getRot(self, a, b): # get rotation matrix from vec a to b
        # a: size 3, b: size 3
        a = a / th.norm(a)
        b = b / th.norm(b)
        nu = th.cross(a, b)
        nu_skew = th.tensor([[0, -nu[2], nu[1]] , 
                             [nu[2], 0, -nu[0]] , 
                             [-nu[1], nu[0], 0] ])
        #s = th.norm(nu)
        c = th.dot(a, b)
        R = th.eye(3) + nu_skew + th.matmul(nu_skew, nu_skew) / (1 + c)
        return R

    def initGeometry(self):
        self.xRange = 1 * np.tan(self.fov/2)
        xStart = 2 * self.xRange * self.bbox[0] - self.xRange
        xEnd   = 2 * self.xRange * (self.bbox[0] + self.bbox[2]) - self.xRange
        self.yRange = float(self.imHeight) / float(self.imWidth) * self.xRange
        yStart = 2 * self.yRange * self.bbox[1] - self.yRange
        yEnd   = 2 * self.yRange * (self.bbox[1] + self.bbox[3]) - self.yRange
        x, y = np.meshgrid(np.linspace(xStart, xEnd, self.res),
                np.linspace(yStart, yEnd, self.res ) )
        y = np.flip(y, axis=0)
        z = -np.ones( (self.res, self.res), dtype=np.float32)
        #pCoord = np.stack([x, y, z]).astype(np.float32)
        #self.pCoord = pCoord[np.newaxis, :, :, :]
        self.pos = th.from_numpy(np.stack([x, y, z], 2).astype(np.float32) )
        self.pos_norm = self.pos.norm(2.0, 2, keepdim=True)
        #self.v = self.getDir().unsqueeze(1)
        self.N = 1
        self.v = self.normalize((self.cameraPos - self.pos).permute(2,0,1).unsqueeze(0).expand(self.N,-1,-1,-1))
        #######
        # tmpX = th.arange(start=int(self.res*self.relC), end=int(self.res*(self.relC+self.relW)), dtype=th.float32).cuda()
        # tmpX = ((tmpX + 0.5) / self.res - 0.5) * self.size
        # tmpY = th.arange(start=int(self.res*self.relR), end=int(self.res*(self.relR+self.relH)), dtype=th.float32).cuda()
        # tmpY = ((tmpY + 0.5) / self.res - 0.5) * self.size
        # y, x = th.meshgrid((tmpY, tmpX))
        # self.pos = th.stack((x, -y, th.zeros_like(x)), 2)
        # self.pos_norm = self.pos.norm(2.0, 2, keepdim=True)

    def initEnv(self):
        Az = ( (np.arange(self.envWidth) + 0.5) / self.envWidth - 0.5 )* 2 * np.pi
        El = ( (np.arange(self.envHeight) + 0.5) / self.envHeight) * np.pi / 2.0
        Az, El = np.meshgrid(Az, El)
        Az = Az.reshape(-1, 1)
        El = El.reshape(-1, 1)
        lx = np.sin(El) * np.cos(Az)
        ly = np.sin(El) * np.sin(Az)
        lz = np.cos(El)
        ls = np.concatenate((lx, ly, lz), axis = 1)

        envWeight = np.sin(El ) * np.pi * np.pi / self.envWidth / self.envHeight

        self.ls = Variable(th.from_numpy(ls.astype(np.float32 ) ) ) # envWidth * envHeight, 3
        self.envWeight = Variable(th.from_numpy(envWeight.astype(np.float32 ) ) )
        self.envWeight = self.envWeight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    def GGX(self, cos_h, alpha):
        c2 = cos_h ** 2
        a2 = alpha ** 2
        den = c2 * a2 + (1 - c2)
        return a2 / (np.pi * den**2 + self.eps)

    def Beckmann(self, cos_h, alpha):
        c2 = cos_h ** 2
        t2 = (1 - c2) / c2
        a2 = alpha ** 2
        return th.exp(-t2 / a2) / (np.pi * a2 * c2 ** 2)

    def Fresnel(self, cos, f0):
        #return f0 + (1 - f0) * (1 - cos)**5
        sphg = th.pow(2.0, ((-5.55473 * cos) - 6.98316) * cos)
        return (1.0 - f0) * sphg

    def Fresnel_S(self, cos, specular):
        sphg = th.pow(2.0, ((-5.55473 * cos) - 6.98316) * cos);
        return specular + (1.0 - specular) * sphg

    def Smith(self, n_dot_v, n_dot_l, alpha):
        def _G1(cos, k):
            return cos / (cos * (1.0 - k) + k)

        k = alpha * 0.5 + self.eps
        return _G1(n_dot_v, k) * _G1(n_dot_l, k)

    def G(self, n_dot_v, n_dot_l, rough):
        def _G1(cos, k):
            return cos / (cos * (1.0 - k) + k)

        k = ( (rough + 1) ** 2 ) / 8.0
        return _G1(n_dot_v, k) * _G1(n_dot_l, k)

    def normalize(self, vec):
        assert(vec.size(0)==self.N)
        assert(vec.size(1)==3)
        assert(vec.size(2)==self.res)
        assert(vec.size(3)==self.res)

        vec = vec / (vec.norm(2.0, 1, keepdim=True))
        return vec

    #def getDir(self, pos):
    def getDir(self):
        # v = self.cameraPos - self.pCoord
        # v = v / np.sqrt(np.maximum(np.sum(v*v, axis=1), 1e-12)[:, np.newaxis, :, :] )
        # v = v.astype(dtype = np.float32)
        # self.v = Variable(torch.from_numpy(v) )
        ######
        #pos = th.from_numpy(self.cameraPos).cuda()
        vec = (self.cameraPos - self.pos).permute(2,0,1).unsqueeze(0).expand(self.N,-1,-1,-1)
        return self.normalize(vec), (vec**2).sum(1, keepdim=True).expand(-1,3,-1,-1)

    def AdotB(self, a, b):
        #ab = (a*b).sum(1, keepdim=True).clamp(min=0).expand(-1,3,-1,-1)
        ab = th.clamp(th.sum(a * b, dim = 2), 0, 1).unsqueeze(2)
        return ab

    #def eval(self, textures, lightPos, cameraPos, light):
    def eval(self, textures, envmapOri, isDemo=False):
        # self.N = textures.size(0)

        if isDemo:
            albedo, normal, rough = textures[0], textures[1], textures[2]
            albedo = th.from_numpy(albedo).cuda()
            normal = th.from_numpy(normal.astype(np.float32)).cuda()
            rough = th.from_numpy(rough).cuda()
        else:
            isSpecular = False
            if textures.size(1) == 9:
                isSpecular = True
                # print('Render Specular\n')

            if isSpecular:
                albedo, normal, rough, specular = tex2map(textures)
            else:
                albedo, normal, rough = tex2map(textures)

        #light = light.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(albedo)
        ldirections = self.ls.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # 1, envWidth * envHeight, 3, 1, 1
        normalPred = th.matmul(self.Rot, normal.permute(0, 2, 3, 1).unsqueeze(-1)).squeeze(-1).permute(0, 3, 1, 2)
        #print(normalPred.shape, self.up.shape) # 1 3 256 256, 3
        camyProj = th.einsum('b,abcd->acd',(self.up, normalPred)).unsqueeze(1).expand_as(normalPred) * normalPred
        camy = th.nn.functional.normalize(self.up.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand_as(camyProj) - camyProj, dim=1)
        camx = -th.nn.functional.normalize(th.cross(camy, normalPred,dim=1), p=1, dim=1)
        l = ldirections[:, :, 0:1, :, :] * camx.unsqueeze(1) \
                + ldirections[:, :, 1:2, :, :] * camy.unsqueeze(1) \
                + ldirections[:, :, 2:3, :, :] * normalPred.unsqueeze(1) # 1, envWidth * envHeight, 3, 1, 1

        normal = normalPred.unsqueeze(1)

        #v, _ = self.getDir(cameraPos)
        #v = self.getDir().unsqueeze(1)
        #l, dist_l_sq = self.getDir(lightPos)
        #h = self.normalize(l + v)
        h = (self.v + l) / 2
        h = h / th.sqrt(th.clamp(th.sum(h*h, dim=2), min = 1e-6).unsqueeze(2) )

        n_dot_v = self.AdotB(normal, self.v)
        n_dot_l = self.AdotB(normal, l)
        n_dot_h = self.AdotB(normal, h)
        v_dot_h = self.AdotB(self.v, h)

        #geom = n_dot_l / dist_l_sq

        D = self.GGX(n_dot_h, rough.unsqueeze(1)**2)
        # D = self.Beckmann(n_dot_h, rough**2)
        isSpecular = False
        if isSpecular:
            F = self.Fresnel_S(v_dot_h, specular)
        else:
            F = self.Fresnel(v_dot_h, self.F0)
        #G = self.Smith(n_dot_v, n_dot_l, rough.unsqueeze(1)**2)
        G = self.G(n_dot_v, n_dot_l, rough.unsqueeze(1))

        # lambert brdf
        f1 = albedo.unsqueeze(1) / np.pi
        if isSpecular:
            f1 *= (1 - specular)
        # cook-torrence brdf
        f2 = D * F * G / (4 * n_dot_v * n_dot_l + self.eps)

        # brdf
        kd = 1; ks = 1
        f = kd * f1 + ks * f2

        # rendering
        #img = f * geom * light
        envmap = th.from_numpy(envmapOri).unsqueeze(0) # [1, 3, 119, 160, 8, 16]
        envR, envC = envmap.size(2), envmap.size(3)
        envmap = envmap.view([1, 3, envR, envC, self.envWidth * self.envHeight ] )
        envmap = envmap.permute([0, 4, 1, 2, 3] ) # [1, self.envWidth * self.envHeight, 3, envR, envC]
        envmap = th.nn.functional.interpolate(envmap.squeeze(0), size=(self.imHeight, self.imWidth) )
        # [self.envWidth * self.envHeight, 3, imHeight, imWidth]
        #print(envmap.size(), int(self.imHeight*self.bbox[1]), int(self.imHeight*(self.bbox[1]+self.bbox[3])))
        #print(int(self.imWidth*self.bbox[0]), int(self.imWidth*(self.bbox[0]+self.bbox[2])) )
        envmap = envmap[:, :, int(self.imHeight*self.bbox[1]):int(self.imHeight*(self.bbox[1]+self.bbox[3])), \
                                int(self.imWidth*self.bbox[0]):int(self.imWidth*(self.bbox[0]+self.bbox[2]))]
        self.envmap = th.nn.functional.interpolate(envmap, size=(self.res, self.res)).unsqueeze(0).cuda()

        img = f * n_dot_l * self.envmap * self.envWeight.expand([1, self.envWidth * self.envHeight, 3, self.res, self.res] )
        img = th.sum(img, dim=1)
        return img.clamp(0,1)

def png2tex(fn, mask_path=None, applyMask2=False):
    png = Image.open(fn)
    if png.height != 256:
        png = png.resize((png.width // png.height * 256, 256))
    png = gyPIL2Array(png)

    res = png.shape[0]
    if applyMask2:
        mask = Image.open(mask_path)
        if mask.height != 256:
            mask = mask.resize((png.width // png.height * 256, 256))
        mask = gyPIL2Array(mask)[:,:,np.newaxis]

        basecolor = png[:,:res,:3]
        def loadMap(fn, cat):
            cat = Image.open(fn.replace('baseColor', cat)).convert('RGB')
            if cat.height != 256:
                cat = cat.resize((cat.width // cat.height * 256, 256))
            cat = gyPIL2Array(cat)
            return cat
        metallic = loadMap(fn, 'metallic')
        normal = loadMap(fn, 'normal') * mask
        rough = loadMap(fn, 'roughness') * mask
        #normal = loadMap(fn, 'normal')
        #rough = loadMap(fn, 'roughness')

        basecolor = basecolor ** (2.2)
        albedo = basecolor * (1-metallic)
        albedo = albedo ** (1/2.2)
        specular = basecolor * metallic + 0.04 * (1-metallic)
        specular = specular ** (1/2.2) * mask
        #specular = specular ** (1/2.2)
        tex = th.cat((th.from_numpy(albedo), th.from_numpy(normal[:,:,0:2]), th.from_numpy(rough[:,:,0]).unsqueeze(2)), 2)
        tex = th.cat((tex,th.from_numpy(specular)),2)

    ##### concatenate flat material maps if only diffuse
    elif png.shape[1] == res:
        # png: [0, 1]
        albedo = png[:,:res,:3]
        if mask_path is not None:
            # Fill in Ref unseen pixels with average color of seen pixels
            mask = Image.open(mask_path)
            if mask.height != 256:
                mask = mask.resize((png.width // png.height * 256, 256))
            mask = gyPIL2Array(mask)[:,:,np.newaxis]
            avgColor = np.sum(albedo * mask, axis=(0, 1), keepdims=True) / np.sum(mask)
            albedo = np.ones_like(albedo) * avgColor * (1-mask) + albedo * mask
        normal = 0.5 * np.ones_like(albedo[:,:,0:2])
        rough = np.ones_like(albedo[:,:,0])
        tex = th.cat((th.from_numpy(albedo), th.from_numpy(normal), th.from_numpy(rough).unsqueeze(2)), 2)
        specular = 0.04 * np.ones_like(albedo)
        tex = th.cat((tex,th.from_numpy(specular)),2)
    #####
    else:
        isSpecular = False
        if png.shape[1]//png.shape[0] == 4:
            isSpecular = True
        albedo = png[:,:res,:]
        normal = png[:,res:res*2,0:2]
        rough = png[:,res*2:res*3,0]
        tex = th.cat((th.from_numpy(albedo), th.from_numpy(normal), th.from_numpy(rough).unsqueeze(2)), 2)
        if isSpecular:
            # print('Specular!!')
            specular = png[:,res*3:res*4,:]
            tex = th.cat((tex,th.from_numpy(specular)),2)
    tex = tex * 2 - 1
    return tex.permute(2,0,1).unsqueeze(0).cuda(), res

def dir2tex(dn):
    def loadAsArray(fn):
        png = Image.open(fn)
        if png.height != 256:
            png = png.resize((png.width // png.height * 256, 256))
        png = gyPIL2Array(png)
        return png
    albedo = loadAsArray(os.path.join(dn, 'diffuse.png'))
    normal = loadAsArray(os.path.join(dn, 'normal.png'))
    rough  = loadAsArray(os.path.join(dn, 'rough.png'))
    tex = th.cat((th.from_numpy(albedo), th.from_numpy(normal[:,:,0:2]), th.from_numpy(rough[:,:]).unsqueeze(2)), 2)
    #tex = th.cat((th.from_numpy(albedo), th.from_numpy(normal[:,:,0:2]), th.from_numpy(rough[:,:,0]).unsqueeze(2)), 2)
    specular = 0.04 * np.ones_like(albedo)
    tex = th.cat((tex,th.from_numpy(specular)),2)
    tex = tex * 2 - 1
    return tex.permute(2,0,1).unsqueeze(0).cuda()

def tex2png(tex, fn, isVertical=False):
    isSpecular = False
    if tex.size(1) == 9:
        isSpecular = True

    if isSpecular:
        albedo, normal, rough, specular = tex2map(tex)
    else:
        albedo, normal, rough = tex2map(tex)

    albedo = gyTensor2Array(albedo[0,:].permute(1,2,0))
    normal = gyTensor2Array((normal[0,:].permute(1,2,0)+1)/2)
    rough  = gyTensor2Array(rough[0,:].permute(1,2,0))

    albedo = gyArray2PIL(gyApplyGamma(albedo, 1/2.2))
    normal = gyArray2PIL(normal)
    rough  = gyArray2PIL(gyApplyGamma(rough, 1/2.2))

    if isVertical:
        png = gyConcatPIL_v(gyConcatPIL_v(albedo,normal), rough)
    else:
        png = gyConcatPIL_h(gyConcatPIL_h(albedo,normal), rough)

    if isSpecular:
        specular = gyTensor2Array(specular[0,:].permute(1,2,0))
        specular = gyArray2PIL(gyApplyGamma(specular, 1/2.2))
        if isVertical:
            png = gyConcatPIL_v(png, specular)
        else:
            png = gyConcatPIL_h(png, specular)

    if fn is not None:
        png.save(fn)
    return png

def renderTex(fn_tex, res, size, lp, cp, L, fn_im):
    # print(lp, cp, L)
    textures, tex_res = png2tex(fn_tex)
    # tex2png(textures, 'a.png')
    # exit()
    if res > tex_res:
        print("[Warning in render.py::renderTex()]: request resolution is larger than texture resolution")
        exit()
    renderObj = Microfacet(res=tex_res, size=size)
    im = renderObj.eval(textures, lightPos=lp, \
        cameraPos=cp, light=th.from_numpy(L).cuda())
    im = gyApplyGamma(gyTensor2Array(im[0,:].permute(1,2,0)), 1/2.2)
    im = gyArray2PIL(im)
    if res < tex_res:
        im = im.resize((res, res), Image.LANCZOS)
    if fn_im is not None:
        im.save(fn_im)
    return im
