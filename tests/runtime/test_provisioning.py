import hashlib,io,json,tarfile,tempfile,tomllib,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
import yaml
from hope_cad.paths import ROOT
from hope_cad.data.transfer import download,sha256
from hope_cad.data.download import prepare,unpack,asset_destination,CATEGORIES,tree_manifest
from hope_cad.cli import parser

class Response(io.BytesIO):
    def __init__(self,data,status=200,headers=None):super().__init__(data);self.status=status;self.headers=headers or {}

class ProvisioningContracts(unittest.TestCase):
    def test_download_range_restart_hash_failure_and_idempotency(self):
        data=b'fixed-test-asset';h=hashlib.sha256(data).hexdigest()
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            for status in (200,206):
                dest=root/str(status);part=dest.with_name(dest.name+'.part');part.write_bytes(data[:4])
                def server(req,**kw):
                    self.assertEqual(req.get_header('Range'),'bytes=4-')
                    return Response(data if status==200 else data[4:],status,{'Content-Range':f'bytes 4-{len(data)-1}/{len(data)}'})
                receipt=download('https://fixture',dest,h,len(data),opener=server)
                self.assertEqual(dest.read_bytes(),data);self.assertEqual(receipt['sha256'],h)
                download('https://fixture',dest,h.upper(),len(data),opener=lambda *a,**k:self.fail('must not re-download'))
                dest.write_bytes(b'wrong')
                with self.assertRaises(ValueError):download('https://fixture',dest,h,len(data))
            dest=root/'bad';dest.with_name('bad.part').write_bytes(data[:4])
            with self.assertRaises(ValueError):download('https://fixture',dest,h,len(data),opener=lambda *a,**k:Response(data[4:],206,{'Content-Range':'bytes 5-8/100'}))
            with self.assertRaises(ValueError):download('https://fixture',root/'corrupt',h,len(data),opener=lambda *a,**k:Response(b'x'*len(data)))
            self.assertFalse((root/'corrupt').exists());self.assertTrue((root/'corrupt.part').exists())
            (root/'owned.download.lock').write_text('owner')
            with self.assertRaises(FileExistsError):download('https://fixture',root/'owned',h,len(data))
            self.assertEqual((root/'owned.download.lock').read_text(),'owner')

    def test_interrupted_download_resumes_and_never_publishes_partial(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'asset';data=b'abcdefgh';h=hashlib.sha256(data).hexdigest()
            class Interrupted(Response):
                def read(self,n=-1):
                    if self.tell():raise OSError('connection lost')
                    return super().read(3)
            with self.assertRaises(OSError):download('https://fixture',path,h,8,opener=lambda *a,**k:Interrupted(data))
            self.assertFalse(path.exists());self.assertEqual((Path(t)/'asset.part').read_bytes(),b'abc')
            download('https://fixture',path,h,8,opener=lambda *a,**k:Response(data[3:],206,{'Content-Range':'bytes 3-7/8'}))
            self.assertEqual(path.read_bytes(),data)

    def test_unpack_rejects_traversal_and_links_before_writes(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);stage=root/'stage';stage.mkdir()
            for name in ('../escape','/absolute','C:/escape'):
                path=root/'bad.zip'
                with zipfile.ZipFile(path,'w') as z:z.writestr(name,b'bad')
                with self.assertRaises(ValueError):unpack(path,stage)
            path=root/'bad.tar'
            with tarfile.open(path,'w') as tf:
                entry=tarfile.TarInfo('link');entry.type=tarfile.SYMTYPE;entry.linkname='../escape';tf.addfile(entry)
            with self.assertRaises(ValueError):unpack(path,stage)
            self.assertEqual(list(stage.iterdir()),[])

    def test_dataset_staged_publish_reuse_and_content_change(self):
        with tempfile.TemporaryDirectory() as t,patch('hope_cad.data.download.ROOT',ROOT.parent/'fixture_source_not_checkout'):
            root=Path(t);archive=root/'data.zip'
            # Tiny contents; full registered counts exercise the actual tree validator.
            with zipfile.ZipFile(archive,'w') as z:
                for c,n in zip(CATEGORIES,[209]+[244]*13+[248]):
                    for i in range(n):z.writestr(f'wrapped/{c}/train/good/{i:03d}.png',b'fixture')
                    for i in range(115):z.writestr(f'wrapped/{c}/test/good/{i:03d}.png',b'fixture')
                    z.writestr(f'wrapped/{c}/ground_truth/',b'')
            output=root/'mvtec'
            with self.assertRaises(ValueError):prepare(output,archive=archive)
            receipt=prepare(output,archive=archive,archive_sha256=sha256(archive))
            self.assertEqual(receipt['tree']['train_normals'],3629);self.assertEqual(receipt['tree']['test_images'],1725)
            self.assertEqual(prepare(output),receipt)
            flat=root/'flat.zip'
            with zipfile.ZipFile(archive) as src,zipfile.ZipFile(flat,'w') as dst:
                for name in src.namelist():dst.writestr(name.removeprefix('wrapped/'),src.read(name))
            flat_receipt=prepare(root/'flat-data',archive=flat,archive_sha256=sha256(flat))
            self.assertEqual(flat_receipt['tree'],receipt['tree'])
            (output/'bottle/train/good/000.png').write_bytes(b'changed')
            with self.assertRaises(ValueError):prepare(output)
            unknown=root/'existing';unknown.mkdir()
            with self.assertRaises(ValueError):prepare(unknown)

    def test_repo_layout_is_ignored_and_not_scientific_source(self):
        for path in (ROOT,ROOT/'src/hope_cad',ROOT/'tests'):
            with self.assertRaises(ValueError):asset_destination(path)
        self.assertEqual(asset_destination(ROOT/'workspace/data/mvtec'),ROOT/'workspace/data/mvtec')
        self.assertEqual(asset_destination(ROOT/'workspace/weights/model.safetensors'),ROOT/'workspace/weights/model.safetensors')
        from hope_cad.runtime.identity import output_directory,source_files
        self.assertEqual(output_directory(ROOT/'runs/one',data_root=ROOT/'workspace/data/mvtec',checkpoint=ROOT/'workspace/weights/vit.safetensors'),ROOT/'runs/one')
        self.assertFalse(any(n.startswith(('workspace/','data/','weights/','runs/','records/')) for n in source_files()))
        args=parser().parse_args(['research','--phase','preflight','--run-dir','runs/one'])
        self.assertEqual(Path(args.data_root),ROOT/'workspace/data/mvtec')
        self.assertEqual(Path(args.checkpoint).parent,ROOT/'workspace/weights')

    def test_pixi_lock_linux_pins_and_editable_install(self):
        manifest=tomllib.loads((ROOT/'pixi.toml').read_text());lock=yaml.safe_load((ROOT/'pixi.lock').read_text())
        self.assertTrue(manifest['pypi-dependencies']['hope-cad']['editable'])
        entries=lock['environments']['default']['packages']['linux-64']
        selected=[next(iter(v.values())) for v in entries]
        for package in ('python-3.13.','numpy-2.1.3-','scikit-learn-1.6.1-','torch-2.11.0','torchvision-0.26.0','pillow-12.3.0','timm-1.0.29','safetensors-0.8.0'):
            self.assertTrue(any(package in n for n in selected),package)
        for name in ('torch','torchvision'):
            self.assertEqual(manifest['target']['linux-64']['pypi-dependencies'][name]['index'],'https://download.pytorch.org/whl/cu128')
        self.assertNotIn('extra-index-urls',manifest['pypi-options'])
        # Lock contains an editable path entry, not a dependency on a sibling checkout.
        self.assertIn({'pypi':'./'},entries)
