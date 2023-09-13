testing = True
if testing:
    import pytest
    from sys import platform

    if platform in ["linux","linux2"]:
        import pytest_xvfb
        @pytest.fixture(autouse=True, scope='session')
        def ensure_xvfb() -> None:
            if not pytest_xvfb.has_executable("Xvfb"):
                raise Exception("Tests need Xvfb to run.")

import pycinema.filters

def test_render():

    resolution = (512,256)
    phi_samples = (20,360,60)
    theta_samples = (20,20,45)
    time_samples = 0.1

    plane_images = pycinema.filters.ShaderDemoScene()
    plane_images.inputs.objects.set((1,0,0),False) # Plane Only
    plane_images.inputs.resolution.set(resolution,False)
    plane_images.inputs.phi_samples.set(phi_samples,False)
    plane_images.inputs.theta_samples.set(theta_samples,False)
    plane_images.inputs.time_samples.set(time_samples,False)
    plane_images.update()

    sphere_images = pycinema.filters.ShaderDemoScene()
    sphere_images.inputs.objects.set((0,1,1),False) # Big and Small Sphere
    sphere_images.inputs.resolution.set(resolution,False)
    sphere_images.inputs.phi_samples.set(phi_samples,False)
    sphere_images.inputs.theta_samples.set(theta_samples,False)
    sphere_images.inputs.time_samples.set(time_samples,False)
    sphere_images.update()

    spheres_colored_by_y = pycinema.filters.ColorMapping()
    spheres_colored_by_y.inputs.channel.set( "y", False )
    spheres_colored_by_y.inputs.map.set( "plasma", False )
    spheres_colored_by_y.inputs.range.set( (0,2), False )
    spheres_colored_by_y.inputs.images.set( sphere_images.outputs.images )

    depth_compositing = pycinema.filters.DepthCompositing()
    depth_compositing.inputs.images_a.set(plane_images.outputs.images, False )
    depth_compositing.inputs.images_b.set(spheres_colored_by_y.outputs.images, False )
    depth_compositing.update()

    ssao = pycinema.filters.ShaderSSAO()
    ssao.inputs.radius.set( 0.1, False )
    ssao.inputs.samples.set( 256, False )
    ssao.inputs.diff.set( 0.5, False )
    ssao.inputs.images.set( depth_compositing.outputs.images )

    image_canny = pycinema.filters.ImageCanny()
    image_canny.inputs.thresholds.set( [50,60], False )
    image_canny.inputs.images.set( depth_compositing.outputs.images )

    color_source = pycinema.filters.ColorSource()
    color_source.inputs.rgba.set((200,0,0,255))

    mask_compositing = pycinema.filters.MaskCompositing()
    mask_compositing.inputs.opacity.set(1.0)
    mask_compositing.inputs.images_a.set(ssao.outputs.images, False )
    mask_compositing.inputs.images_b.set(color_source.outputs.rgba, False )
    mask_compositing.inputs.masks.set(image_canny.outputs.images, False )
    mask_compositing.inputs.mask_channel.set('canny')

    annotation = pycinema.filters.Annotation()
    annotation.inputs.color.set( (200,200,200), False )
    annotation.inputs.size.set( 14, False )
    annotation.inputs.xy.set( (10,10), False )
    annotation.inputs.spacing.set( 10, False )
    annotation.inputs.images.set( mask_compositing.outputs.images )

    border = pycinema.filters.ImageBorder()
    border.inputs.color.set( (0,140,140,255) )
    border.inputs.width.set( 5 )
    border.inputs.images.set( annotation.outputs.images )

    # Test Output
    if testing:
        GT = [[139.40163230895996, 0, 255, 0.18886437, 0.4082454], [139.6744270324707, 0, 255, 0.18886437, 0.4082454], [139.56921768188477, 0, 255, 0.18886437, 0.4082454], [139.52686882019043, 0, 255, 0.18886437, 0.4082454], [139.79484176635742, 0, 255, 0.18886437, 0.4082454], [139.57610511779785, 0, 255, 0.18886438, 0.4082454]]
    else:
        import PIL
        GT = []

    def compare(a,b,tolerance):
        for i in range(0,len(a)):
            if abs(a[i]-b[i])>tolerance[i]:
                return False
        return True

    images = border.outputs.images.get()
    s = []
    for i in range(len(images)):
        image = images[i]
        data = [
            image.channels['rgba'].mean(),
            image.channels['rgba'].min(),
            image.channels['rgba'].max(),
            image.channels['depth'].min(),
            image.channels['depth'].max()
        ]
        s.append(data)
        if not testing:
            display(PIL.Image.fromarray(image.channels['rgba']))
        elif not compare(s[i],GT[i],[0.2,0.2,0.2,0.05,0.05]):
            print(str(data))
            print(GT[i])
            raise ValueError('Generated Data does not correspond to Ground Truth')

    if not testing:
        print(s)

    print("Test Complete")

test_render()
