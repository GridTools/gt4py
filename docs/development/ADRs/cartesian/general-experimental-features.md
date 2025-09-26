# Experimental features

In the context of DSL development, facing long times from initial idea to fully-fledge feature, we decided to allow experimental features to gather early feedback from users on real-world codes. We considered only releasing fully-fledge features and accept that experimental features might change, be replaced or never make it to fully-fledge features.

## Context

Writing a good domain specific language (DSL) is hard. Generally speaking one tries to be as expressive as possible while limiting the surface area of the DSL to a minimum. Features increasing the DSL surface thus need careful and deliberate design decisions. On the other hand, testing with real-world use-cases can be paramount to finding a tailored DSL representation. To accelerate this feedback cycle, we propose to iterate quicker by releasing experimental features to gather feedback, which will help to fully flesh out the feature.

## Decision

We decided to introduce so called "experimental features", a preview of features that we think will be useful to our users.

Experimental features allow us to gather early feedback on real-world use-cases directly from our users.

To ensure that users, who choose not to use experimental features, aren't impacted negatively in any way by their existence, we impose the following guidelines:

1. Experimental features must be additive.
2. Usage of experimental features must be strictly optional and on an opt-in basis.
3. We reserve the right to modify, replace, and/or remove experimental features without prior notice (e.g. no deprecation periods even for breaking changes).
4. Experimental features need to work in the [`debug` backend](./backend-debug.md) - after all, this is what we built it for.

## Consequences

What is now easier to do? What becomes more difficult with this change?

With experimental features, we get fast feedback from our users. Our users get early access to (potential) features. We all get a more collaborative and iterative approach on the future development of `gt4py.cartesian`.

We are aware (and communicate clearly) that experimental features

- might break in unforeseen ways
- might not be supported in all backends
- might change at any point in time without prior warning, even in breaking ways
- might be replaced with a more suitable feature or be entirely removed without replacements.

## Alternatives considered

### Not having it

Not having experimental features equals to keeping the status quo. Pros and cons are as outlined in this document above.

### Reducing the number of backends

One could argue that shipping fully-fledged features is only expensive / time-consuming because there are many backends where the feature would need to be implemented. This is only partially true. While implementing a feature only in specific backends does save time, the point of experimental features isn't per se to ship faster. It's about getting feedback from the community on real-world use-cases. If a feature is well-designed and understood / used by the community as expected / intended can be studied as soon as the feature is implemented in one backend. This feedback could then be leveraged to refine that preliminary implementation. Once a good design is found, the feature can be spread to other backends.

## References

ADR / motivation for the [debug backend](./backend-debug.md) as a playground for new/experimental DSL features.
