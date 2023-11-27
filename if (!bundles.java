        if (!bundles.isEmpty()) {
            List<Long> bundleProductTypIds = bundles.stream()
                    .findFirst().get().getOptions().stream()
                    .flatMap(o -> o.getContents().stream()
                            .map(s -> s.productType().getId()))
                    .toList();
        }