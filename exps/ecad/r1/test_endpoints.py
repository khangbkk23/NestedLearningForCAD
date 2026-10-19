import unittest
import numpy as np
from .endpoints import upper_tail,old_tail,acquisition,benefit,acquisition_regression,whole_image_draws,simultaneous_basic,pre_return_risk


class Endpoints(unittest.TestCase):
    def test_nested_tail(self):
        a=np.zeros((16,784));b=a.copy();b[-1,-98:]=2
        self.assertEqual(old_tail(a,b,2)['positive'],.5)
        self.assertEqual(old_tail(a,-b,2)['positive'],0)
        self.assertEqual(float(upper_tail(np.arange(8))),7)

    def test_sign_and_zero(self):
        self.assertGreater(benefit(.2,.1)['relative'],0)
        self.assertLess(benefit(.1,.2)['relative'],0)
        self.assertIsNone(benefit(0,0)['relative'])
        self.assertEqual(benefit(0,2)['absolute'],-2)
        self.assertEqual(benefit(.019,.001)['status'],'NO_HEADROOM')
        self.assertIsNone(acquisition_regression(0,1)['relative'])

    def test_acquisition_is_absolute(self):
        d=np.full((16,784),3.)
        self.assertEqual(acquisition(d,2),{'mean':1.5,'q95':1.5})
        self.assertEqual(acquisition_regression(2,3)['relative'],.5)

    def test_whole_image_pairing_and_family(self):
        draws=list(whole_image_draws({'a':16,'b':16},draws=4))
        again=list(whole_image_draws({'a':16,'b':16},draws=4))
        for x,y in zip(draws,again):np.testing.assert_equal(x,y)
        p=np.array([.1,-.2]);r=np.array([[.11,-.19],[.12,-.22],[.09,-.2]])
        ci=simultaneous_basic(p,r)
        np.testing.assert_allclose(ci,[[.08,.12],[-.22,-.18]])

    def test_max_before_order_average(self):
        def entry(x):return {'acquisition':np.zeros((16,784)),'current':np.full((16,784),x),'scale':1}
        orders=[{'a':entry(2),'b':entry(0)},{'a':entry(0),'b':entry(2)}]
        self.assertEqual(pre_return_risk(orders),2)


if __name__=='__main__':unittest.main(verbosity=2)
