import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Getting Started',
      collapsible: false,
      items: [
        'getting-started/installation',
        'getting-started/quick-start',
        'getting-started/configuration-wizard',
      ],
    },
    {
      type: 'category',
      label: 'Core Reference',
      collapsible: false,
      items: [
        'data-format',
        'config-format',
        'recipes',
        'evaluation-results',
        'report-formats',
      ],
    },
    {
      type: 'category',
      label: 'Guides',
      collapsible: false,
      items: [
        'optimizers',
        'field-metrics',
        'transformer-models',
        'self-hosted-models',
      ],
    },
    {
      type: 'category',
      label: 'Examples',
      collapsible: true,
      link: {
        type: 'doc',
        id: 'examples/index',
      },
      items: [
        'examples/sentiment-classification',
        'examples/affiliation-extraction',
        'examples/transformer-comparison',
        'examples/multimodal-molecules',
        'examples/incremental-evaluation',
      ],
    },
  ],
};

export default sidebars;
