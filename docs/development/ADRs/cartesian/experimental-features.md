# Experimental features

In the context of DSL development, facing long times from initial idea to fully-fledge feature, we decided to allow experimental features to gather early feedback from users on real-world codes. We considered keeping experimental features on branches and accept that experimental features might change, be replaced or never make it to fully-fledge features.

## Context

Writing a good domain specific language (DSL) is hard. Generally speaking, one tries to be as expressive as possible while limiting the surface area of the DSL to a minimum. Features increasing the DSL surface thus need careful and deliberate design decisions. On the other hand, testing with real-world use-cases can be paramount to finding a tailored DSL representation. Feedback from domain experts is thus an important part of building a DSL that fits the domain and that domain scientist are happy to work with. To accelerate this feedback cycle, we propose to release "experimental features" to gather feedback on new features early, which will help to fully flesh out the feature. Not every feature can be released as experimental feature. We will impose a set of guidelines to ensure smooth operations.

## Decision

We decided to introduce so called "experimental features", a preview of features that we think will be useful to our users. Doing so, allows us to gather feedback early and from on real-world use-cases directly from our users.

Experimental features have to follow a set of rules to ensure they don't disturb normal operations. We thus impose the following guidelines:

1. Experimental features must be additive.
2. Usage of experimental features must be strictly optional and on an opt-in basis.
3. Experimental features must log a one-time warning on first usage/occurrence.
4. We reserve the right to modify, replace, and/or remove experimental features without prior notice (e.g. no deprecation periods even for breaking changes) at any point in time.
5. Experimental features need to work in the [`debug` backend](./backend/debug.md) - after all, this is what we built it for.

## Consequences

With experimental features, we get fast feedback from our users. Our users get early access to (potential) future features. And we all get a more collaborative and iterative approach on the future development of `gt4py.cartesian`.

We are aware and communicate clearly that experimental features

- might break science code in unforeseen ways,
- might not be supported in all backends,
- might change at any point in time without prior warning, even in breaking ways,
- and might be replaced with a more suitable (set of) feature(s) or be entirely removed without replacements.

## Alternatives considered

### Keep experimental features on branches

Feature development happens on (feature) branches. Wouldn't it be easier to keep experimental features on branches until they are ready? Yes and No.

For some features, especially ones that aren't additive and/or change a default, working on a branch until the feature is fully-fleshed out is the only option. So feature development on feature branches is not going away.

Experimental features can help in situations where multiple (additive) features play together e.g. to allow a new workflow. It is possible to maintain a branch combining the `main` branch and all the necessary experimental features for that new workflow. This "experimental branch" not only creates overhead on our side maintaining that branch, it also increases friction for our users to test the new workflow. Users trying the new workflow will have to re-install GT4Py and switch between versions when going back and forth between the old and new way of doing things. If the new workflow can be provided by a series of experimental features, users can seamlessly switch between workflows. This lowers the bar for users to try out the new feature(s), which contributes to faster feedback cycles, which ultimately allows us to build a DSL that users like.

### Reducing the number of backends

One could argue that shipping fully-fledged features is only expensive / time-consuming because there are (too) many backends where the feature would need to be implemented. This is only partially true. While implementing a feature only in specific backends does save time, the point of experimental features isn't per se to ship faster. It's about getting feedback from the community on real-world use-cases. Whether a feature is well-designed and understood / used by the community as expected / intended can be studied as soon as the feature is implemented in one backend. This feedback can then be leveraged to refine that preliminary implementation. Once a good design is found, the feature can be spread to other backends.

## References

ADR / motivation for the [debug backend](../backend/debug.md) as a playground for new/experimental DSL features.
