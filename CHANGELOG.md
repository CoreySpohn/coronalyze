# Changelog

## [1.2.0](https://github.com/CoreySpohn/coronalyze/compare/v1.1.1...v1.2.0) (2026-07-18)


### Features

* **api:** export seam contract types at top level ([2d843a4](https://github.com/CoreySpohn/coronalyze/commit/2d843a4487dd4eb3ccb3b231abc9bba023848a4d))
* **api:** export the composable detection core and matched filter ([49f83b1](https://github.com/CoreySpohn/coronalyze/commit/49f83b145fccd1f8113578e206e631e93710b5ef))
* **api:** export the template-filter surface ([6d0c2ca](https://github.com/CoreySpohn/coronalyze/commit/6d0c2ca5a1984b26148dd7f61df2281c4d9c6953))
* **contracts:** DetectionStats seam output type ([9da923d](https://github.com/CoreySpohn/coronalyze/commit/9da923d8954e646c5df963a91901d07323542eff))
* **contracts:** FrameSet seam input type ([8d066d1](https://github.com/CoreySpohn/coronalyze/commit/8d066d184914bd615a5e68cc735eaabb365bf5af))
* **detection:** aperture and annulus noise samplers with center support ([35e62f5](https://github.com/CoreySpohn/coronalyze/commit/35e62f585fef2cf0fc583f1dde924ad6071a3517))
* **detection:** DetectionEstimator with golden parity against the fused cores ([5e82508](https://github.com/CoreySpohn/coronalyze/commit/5e82508925df33553a252f50142308bcbe6dfe83))
* **detection:** differentiable fixed-size patch extraction ([112b37f](https://github.com/CoreySpohn/coronalyze/commit/112b37f7824db5714f58a7a32f72cbc3664b5214))
* **detection:** filter/sampler/test ABCs and the aperture and Gaussian filters ([fc98eb8](https://github.com/CoreySpohn/coronalyze/commit/fc98eb8cbe97eb3f4ff69fdeed58282d3dc70642))
* **detection:** PSF-template matched filter with per-candidate template binding ([17ef11d](https://github.com/CoreySpohn/coronalyze/commit/17ef11d6262c6f4c496de67be78c49136745827c))
* **detection:** two-sample t, annulus sigma, and Grubbs significance tests ([149363b](https://github.com/CoreySpohn/coronalyze/commit/149363b56da6143b21351149698cd0a8ad6be436))
* **geometry:** traceable n_reference_apertures, calculate_n_apertures delegates ([01e36ec](https://github.com/CoreySpohn/coronalyze/commit/01e36ec1e593ee96faf14a81a7bc5ff48f22cd9d))
* **pipelines:** calculate_yield_snr accepts an optional detection estimator ([17b1faa](https://github.com/CoreySpohn/coronalyze/commit/17b1faa554e54c998d4a98320fcd23b005161ee1))
* **postproc:** AbstractPostProcessing seam ABC and Mawet arm ([a6823d5](https://github.com/CoreySpohn/coronalyze/commit/a6823d56036ad6454681f2010ab021fad6e0e80e))
* **postproc:** Mawet arm consults FrameSet.center_yx via the composed estimator ([f126175](https://github.com/CoreySpohn/coronalyze/commit/f126175306814f482eda8a7db9e241b30db82da9))
* **postproc:** PSF-template matched-filter arm with inverse-variance whitening ([ef0febf](https://github.com/CoreySpohn/coronalyze/commit/ef0febf2f7e5abc330e2a51359eb7848f6c99535))
* **snr:** expose reference-aperture counts via snr_and_dof ([a00fe6d](https://github.com/CoreySpohn/coronalyze/commit/a00fe6d23d607e5ca4afb27f1eacee9f1fbeb516))
* **statistics:** normal and Grubbs survival functions, nan-masked stats ([e8ca00d](https://github.com/CoreySpohn/coronalyze/commit/e8ca00d2278f0d4ca42dd36342317e55d4d2e37a))
* **statistics:** student-t survival function for FPF computation ([08159a6](https://github.com/CoreySpohn/coronalyze/commit/08159a6d1651961cef4cba18f1b0027b81dc4e3f))
* **templates:** template provider interface and precomputed-array provider ([fb4c4f9](https://github.com/CoreySpohn/coronalyze/commit/fb4c4f983ff672482dc60cd0ce56261e39f9185f))
* **templates:** yippy off-axis PSF template provider behind the yippy extra ([63c9114](https://github.com/CoreySpohn/coronalyze/commit/63c9114409b13e0c8ab9da39deb3aa0cfed7b13b))


### Bug Fixes

* **pipelines:** reject MatchedFilterSNREstimator in calculate_yield_snr ([74c0a04](https://github.com/CoreySpohn/coronalyze/commit/74c0a0433ee80a5bcbcc902d2e2dfe011a93547b))
* **statistics:** exact-zero Grubbs FPF at the attainable maximum ([7aa24f2](https://github.com/CoreySpohn/coronalyze/commit/7aa24f20548bc3757909ca4163821619932167fa))

## [1.1.1](https://github.com/CoreySpohn/coronalyze/compare/v1.1.0...v1.1.1) (2026-05-26)


### Bug Fixes

* upstream API updates ([41b04d5](https://github.com/CoreySpohn/coronalyze/commit/41b04d58f1dbaa4a66da57a9c16a3353a8189bc6))

## [1.1.0](https://github.com/CoreySpohn/coronalyze/compare/v1.0.4...v1.1.0) (2026-04-23)


### Features

* **pp:** add PPConfig post-processing module ([5568d51](https://github.com/CoreySpohn/coronalyze/commit/5568d51a39ba68e5d1f55289f18920bc71479014))

## [1.0.4](https://github.com/CoreySpohn/coronalyze/compare/v1.0.3...v1.0.4) (2026-01-27)


### Bug Fixes

* Clean up yippy call ([49b3a8a](https://github.com/CoreySpohn/coronalyze/commit/49b3a8a0def1278a4041c50e68c70268d8be688a))
* Cleaning out unnecessary dependencies ([ad20c4f](https://github.com/CoreySpohn/coronalyze/commit/ad20c4fcee093233792f7b7ff7de56077266c3c4))
* Mask out points in the snr_map at small separations where the results are undefined ([d094c88](https://github.com/CoreySpohn/coronalyze/commit/d094c88b71821bcf90a2a801adbf37ec0d069c8a))
* Updating tests to fix error hopefully ([7c8e139](https://github.com/CoreySpohn/coronalyze/commit/7c8e1399ae7789bd58638f3574cb29e6dcf8b91e))

## [1.0.3](https://github.com/CoreySpohn/coronalyze/compare/v1.0.2...v1.0.3) (2026-01-27)


### Bug Fixes

* Version management fix ([35f9175](https://github.com/CoreySpohn/coronalyze/commit/35f9175fc6699c684e5939f68454fbc3184ea1ce))

## [1.0.2](https://github.com/CoreySpohn/coronalyze/compare/v1.0.1...v1.0.2) (2026-01-27)


### Bug Fixes

* Revert name back to coronalyze ([8dc5d93](https://github.com/CoreySpohn/coronalyze/commit/8dc5d9368db3dfeb69e7b270c9f6fb959715d5fe))

## [1.0.1](https://github.com/CoreySpohn/coronalyze/compare/v1.0.0...v1.0.1) (2026-01-26)


### Bug Fixes

* Update the wheel targets to avoid data files ([cd4299d](https://github.com/CoreySpohn/coronalyze/commit/cd4299d0e12045fdb64f7418f63ee0b4bdffe269))

## 1.0.0 (2026-01-26)


### Features

* Add literally everything ([60a8394](https://github.com/CoreySpohn/coronalyze/commit/60a83946f340c26c99d503174a11eebd2f7ae434))
* Introduce the `coronalyze` library for JAX-based coronagraphic image post-processing, including aperture photometry and SNR calculations, along with project infrastructure. ([053595f](https://github.com/CoreySpohn/coronalyze/commit/053595fc5936025493872363e35ae683a4561bde))
